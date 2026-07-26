// React hook: retrieves evidence, streams grounded synthesis, and builds the KG.

import { useCallback, useRef, useState } from "react";
import {
  ApiError,
  fetchSubgraph,
  fetchSynthesisStream,
  matchQuery,
} from "./api";
import type {
  ActiveDocumentContext,
  BackendEmptyOutcome,
  BackendSynthesisStreamEvent,
  RetrievalContext,
} from "./api";
import {
  adaptResult,
  adaptGraphFromMatch,
  adaptSubgraphNode,
  adaptSubgraphLink,
} from "./adapters";
import type { TrialResult, GraphNode, GraphEdge } from "./adapters";

export type QueryState = "idle" | "loading" | "results" | "error" | "empty";

interface QueryMeta {
  latencyMs: number;
  indexVersion: string;
  strategy: string;
  totalHits: number;
}

interface UseQueryPollResult {
  queryState: QueryState;
  results: TrialResult[];
  graph: { nodes: GraphNode[]; edges: GraphEdge[] } | null;
  meta: QueryMeta | null;
  errorMsg: string | null;
  emptyOutcome: BackendEmptyOutcome | null;
  synthesis: string | null;
  synthesisLoading: boolean;
  synthesisError: ApiError | null;
  synthesisModel: string | null;
  synthesisFallbackUsed: boolean;
  runQuery: (query: string, options?: RunQueryOptions) => void;
  retrySynthesis: () => void;
  resetQuery: () => void;
}

interface RunQueryOptions {
  topK?: number;
  activeDocument?: ActiveDocumentContext | null;
}

interface SynthesisRequest {
  query: string;
  evidenceId: string;
  requestId: number;
}

const DEFAULT_EMPTY_OUTCOME: BackendEmptyOutcome = {
  code: "no_relevant_evidence",
  message: "No relevant evidence was found.",
  action: "Use more specific clinical terms or verify the selected source.",
};

function toApiError(error: unknown): ApiError {
  if (error instanceof ApiError) return error;
  return new ApiError(0, {
    code: null,
    message: "Synthesis could not be completed. Please try again.",
    retryable: true,
  });
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function parseSseFrame(frame: string): BackendSynthesisStreamEvent | null {
  const data = frame
    .split(/\r?\n/)
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trim())
    .join("");
  if (!data) return null;

  let payload: unknown;
  try {
    payload = JSON.parse(data);
  } catch {
    throw new ApiError(0, {
      code: "invalid_stream_event",
      message: "The synthesis stream returned malformed data.",
      retryable: true,
    });
  }
  if (!isRecord(payload) || typeof payload.type !== "string") {
    throw new ApiError(0, {
      code: "invalid_stream_event",
      message: "The synthesis stream returned an invalid event.",
      retryable: true,
    });
  }

  if (payload.type === "token" && typeof payload.text === "string") {
    return { type: "token", text: payload.text };
  }
  if (payload.type === "meta") {
    return {
      type: "meta",
      model: typeof payload.model === "string" ? payload.model : null,
      fallbackUsed: payload.fallbackUsed === true,
    };
  }
  if (payload.type === "done") return { type: "done" };
  if (payload.type === "error" && typeof payload.message === "string") {
    return {
      type: "error",
      code: typeof payload.code === "string" ? payload.code : null,
      message: payload.message,
      retryable: payload.retryable === true,
    };
  }
  throw new ApiError(0, {
    code: "invalid_stream_event",
    message: "The synthesis stream returned an unsupported event.",
    retryable: true,
  });
}

export function useQueryPoll(): UseQueryPollResult {
  const [queryState, setQueryState] = useState<QueryState>("idle");
  const [results, setResults] = useState<TrialResult[]>([]);
  const [graph, setGraph] = useState<{ nodes: GraphNode[]; edges: GraphEdge[] } | null>(null);
  const [meta, setMeta] = useState<QueryMeta | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [emptyOutcome, setEmptyOutcome] = useState<BackendEmptyOutcome | null>(null);
  const [synthesis, setSynthesis] = useState<string | null>(null);
  const [synthesisLoading, setSynthesisLoading] = useState(false);
  const [synthesisError, setSynthesisError] = useState<ApiError | null>(null);
  const [synthesisModel, setSynthesisModel] = useState<string | null>(null);
  const [synthesisFallbackUsed, setSynthesisFallbackUsed] = useState(false);

  const queryAbortRef = useRef<AbortController | null>(null);
  const synthesisAbortRef = useRef<AbortController | null>(null);
  const requestIdRef = useRef(0);
  const synthesisAttemptRef = useRef(0);
  const synthesisRequestRef = useRef<SynthesisRequest | null>(null);

  const runSynthesis = useCallback(async (request: SynthesisRequest) => {
    synthesisAbortRef.current?.abort();
    const abort = new AbortController();
    synthesisAbortRef.current = abort;
    const attemptId = ++synthesisAttemptRef.current;
    const isCurrent = () => (
      requestIdRef.current === request.requestId
      && synthesisAttemptRef.current === attemptId
      && !abort.signal.aborted
    );

    setSynthesisLoading(true);
    setSynthesisError(null);
    setSynthesis("");
    setSynthesisModel(null);
    setSynthesisFallbackUsed(false);

    try {
      const response = await fetchSynthesisStream(
        request.query,
        request.evidenceId,
        abort.signal,
      );
      if (!response.body) {
        throw new ApiError(0, {
          code: "missing_stream",
          message: "The synthesis response did not include a stream.",
          retryable: true,
        });
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let completed = false;

      while (true) {
        const { done, value } = await reader.read();
        buffer += decoder.decode(value, { stream: !done });
        const frames = buffer.split(/\r?\n\r?\n/);
        buffer = frames.pop() ?? "";

        for (const frame of frames) {
          const event = parseSseFrame(frame);
          if (!event || !isCurrent()) continue;
          if (event.type === "token") {
            setSynthesis((current) => `${current ?? ""}${event.text}`);
          } else if (event.type === "meta") {
            setSynthesisModel(event.model);
            setSynthesisFallbackUsed(event.fallbackUsed);
          } else if (event.type === "error") {
            throw new ApiError(0, {
              code: event.code,
              message: event.message,
              retryable: event.retryable,
            });
          } else {
            completed = true;
          }
        }

        if (done) break;
      }

      if (buffer.trim()) {
        const event = parseSseFrame(buffer);
        if (event?.type === "done") completed = true;
      }
      if (!completed && isCurrent()) {
        throw new ApiError(0, {
          code: "incomplete_stream",
          message: "The synthesis stream ended before completion.",
          retryable: true,
        });
      }
    } catch (error) {
      if (!isCurrent()) return;
      setSynthesisError(toApiError(error));
    } finally {
      if (requestIdRef.current === request.requestId && synthesisAttemptRef.current === attemptId) {
        setSynthesisLoading(false);
      }
    }
  }, []);

  const runQuery = useCallback(async (query: string, options: RunQueryOptions = {}) => {
    queryAbortRef.current?.abort();
    synthesisAbortRef.current?.abort();
    const abort = new AbortController();
    queryAbortRef.current = abort;
    const requestId = ++requestIdRef.current;
    synthesisAttemptRef.current += 1;
    synthesisRequestRef.current = null;

    const activeDocument = options.activeDocument ?? null;
    const context: RetrievalContext = {
      target: activeDocument ? "patient_context" : "literature",
      activeDocument,
    };

    setQueryState("loading");
    setResults([]);
    setGraph(null);
    setMeta(null);
    setErrorMsg(null);
    setEmptyOutcome(null);
    setSynthesis(null);
    setSynthesisLoading(false);
    setSynthesisError(null);
    setSynthesisModel(null);
    setSynthesisFallbackUsed(false);

    try {
      const res = await matchQuery(
        query,
        context,
        options.topK ?? 10,
        abort.signal,
      );
      if (abort.signal.aborted) return;

      if (!res.found) {
        setQueryState("empty");
        setEmptyOutcome(res.empty ?? DEFAULT_EMPTY_OUTCOME);
        setMeta({
          latencyMs: res.latency_ms,
          indexVersion: "live",
          strategy: "GraphRAG",
          totalHits: 0,
        });
        return;
      }
      if (!res.evidenceId) {
        throw new Error("The retrieval service did not return an evidence identifier.");
      }

      const adapted = res.matches.map(adaptResult);
      setResults(adapted);
      setMeta({
        latencyMs: res.latency_ms,
        indexVersion: "live",
        strategy: "GraphRAG + Dense",
        totalHits: adapted.length,
      });
      setQueryState("results");
      const synthesisRequest = {
        query,
        evidenceId: res.evidenceId,
        requestId,
      };
      synthesisRequestRef.current = synthesisRequest;

      const inlineGraph = adaptGraphFromMatch(res.graphFacts, res.matches);
      if (inlineGraph.nodes.length > 0) {
        setGraph(inlineGraph);
      }

      void runSynthesis(synthesisRequest);

      if (res.graphAnchor) {
        void fetchSubgraph(res.graphAnchor, context, abort.signal)
          .then((sub) => {
            if (abort.signal.aborted || requestIdRef.current !== requestId || sub.nodes.length === 0) return;
            const nodes = sub.nodes.map((node, index) => (
              adaptSubgraphNode(node, index, sub.nodes.length)
            ));
            const edges = sub.links.map(adaptSubgraphLink);
            setGraph({ nodes, edges });
          })
          .catch(() => { /* Neo4j enrichment is non-fatal. */ });
      }
    } catch (error) {
      if (abort.signal.aborted) return;
      setQueryState("error");
      setErrorMsg(error instanceof Error ? error.message : String(error));
    }
  }, [runSynthesis]);

  const retrySynthesis = useCallback(() => {
    const request = synthesisRequestRef.current;
    if (request) void runSynthesis(request);
  }, [runSynthesis]);

  function resetQuery() {
    queryAbortRef.current?.abort();
    synthesisAbortRef.current?.abort();
    requestIdRef.current += 1;
    synthesisAttemptRef.current += 1;
    synthesisRequestRef.current = null;
    setQueryState("idle");
    setResults([]);
    setGraph(null);
    setMeta(null);
    setErrorMsg(null);
    setEmptyOutcome(null);
    setSynthesis(null);
    setSynthesisLoading(false);
    setSynthesisError(null);
    setSynthesisModel(null);
    setSynthesisFallbackUsed(false);
  }

  return {
    queryState,
    results,
    graph,
    meta,
    errorMsg,
    emptyOutcome,
    synthesis,
    synthesisLoading,
    synthesisError,
    synthesisModel,
    synthesisFallbackUsed,
    runQuery,
    retrySynthesis,
    resetQuery,
  };
}
