import { useState, type FormEvent } from "react";
import { Button, Card, FormGroup, InputGroup, Intent } from "@blueprintjs/core";
import { login } from "../lib/api";

interface LoginPageProps {
  onSuccess: () => void;
}

export default function LoginPage({ onSuccess }: LoginPageProps) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError]       = useState<string | null>(null);
  const [loading, setLoading]   = useState(false);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      await login(username, password);
      onSuccess();
    } catch {
      setError("Incorrect username or password.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        height: "100vh",
        background: "var(--surface-0)",
      }}
    >
      <Card
        style={{ width: 360, padding: "2rem" }}
        elevation={2}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "0.6rem", marginBottom: "1.5rem" }}>
          {/* Rod of Asclepius — universal medical / HIPAA symbol */}
          <span style={{ fontSize: "1.5rem", lineHeight: 1 }} aria-hidden="true">⚕</span>
          <h2
            style={{
              margin: 0,
              fontSize: "1.1rem",
              fontWeight: 600,
              color: "var(--text-primary)",
              letterSpacing: "0.02em",
            }}
          >
            Clinical Search Sign In
          </h2>
        </div>

        <form onSubmit={handleSubmit} noValidate>
          <FormGroup label="Username" labelFor="login-username">
            <InputGroup
              id="login-username"
              type="text"
              autoComplete="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              disabled={loading}
              autoFocus
            />
          </FormGroup>

          <FormGroup label="Password" labelFor="login-password">
            <InputGroup
              id="login-password"
              type="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              disabled={loading}
            />
          </FormGroup>

          {error && (
            <p
              style={{
                margin: "0 0 1rem",
                fontSize: "0.85rem",
                color: "var(--red-text, #c23030)",
              }}
              role="alert"
            >
              {error}
            </p>
          )}

          <Button
            type="submit"
            intent={Intent.PRIMARY}
            loading={loading}
            fill
            text="Sign In"
          />
        </form>
      </Card>
    </div>
  );
}
