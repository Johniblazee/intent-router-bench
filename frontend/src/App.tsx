import { useEffect, useRef, useState } from "react";
import { Loader2, Send, Zap } from "lucide-react";
import { Badge, Button, Card, Input, Select } from "./components/ui";

type Entry = {
  message: string;
  intent?: string;
  ms?: number;
  loadS?: number;
  response?: string;
  model: string;
  error?: string;
};

export default function App() {
  const [models, setModels] = useState<string[]>([]);
  const [demos, setDemos] = useState<string[]>([]);
  const [model, setModel] = useState("bge-small");
  const [groqKey, setGroqKey] = useState(() => localStorage.getItem("groq_key") ?? "");
  const [geminiKey, setGeminiKey] = useState(() => localStorage.getItem("gemini_key") ?? "");
  const [providerModels, setProviderModels] = useState<string[]>([]);
  const [providerModel, setProviderModel] = useState("");
  const [provErr, setProvErr] = useState("");
  const [fetching, setFetching] = useState(false);
  const [warming, setWarming] = useState(false);
  const [warmed, setWarmed] = useState<Record<string, number> | null>(null);
  const provider = model.startsWith("groq") ? "groq" : model.startsWith("gemini") ? "gemini" : "";

  useEffect(() => {
    setProviderModels([]);
    setProviderModel("");
    setProvErr("");
  }, [provider]);

  async function fetchProviderModels() {
    setFetching(true);
    setProvErr("");
    try {
      const r = await fetch("/api/provider-models", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider, api_key: (provider === "groq" ? groqKey : geminiKey) || null }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(d.detail ?? r.statusText);
      setProviderModels(d.models);
    } catch (e) {
      setProvErr(String(e));
    } finally {
      setFetching(false);
    }
  }
  const [input, setInput] = useState("");
  const [history, setHistory] = useState<Entry[]>([]);
  const [busy, setBusy] = useState(false);
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch("/api/models")
      .then((r) => r.json())
      .then((d) => {
        setModels(d.models);
        setDemos(d.demos);
        setModel((m) => (d.models.includes(m) ? m : d.models[0]));
      });
  }, []);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" });  // braces matter: implicit return becomes React cleanup
  }, [history, busy]);

  async function route(message: string) {
    if (!message.trim() || busy) return;
    setInput("");
    setBusy(true);
    try {
      const apiKey = provider === "groq" ? groqKey : provider === "gemini" ? geminiKey : "";
      const label = providerModel ? `${model} → ${providerModel}` : model;
      const r = await fetch("/api/route", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message,
          model,
          api_key: apiKey || null,
          provider_model: providerModel || null,
        }),
      });
      const d = await r.json();
      const detail = typeof d.detail === "string" ? d.detail : JSON.stringify(d.detail ?? r.statusText);
      setHistory((h) => [
        ...h,
        r.ok
          ? { message, intent: d.intent, ms: d.ms, loadS: d.load_s, response: d.response, model: label }
          : { message, model: label, error: detail },
      ]);
    } catch (e) {
      setHistory((h) => [...h, { message, model, error: String(e) }]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex h-screen">
      <aside className="flex w-80 flex-col gap-4 border-r border-border p-4">
        <div className="flex items-center gap-2">
          <Zap className="size-5" />
          <h1 className="text-lg font-semibold">Intent Router Test</h1>
        </div>
        <div className="space-y-1.5">
          <Button
            className="w-full"
            disabled={warming}
            onClick={async () => {
              setWarming(true);
              try {
                const r = await fetch("/api/preload", { method: "POST" });
                setWarmed((await r.json()).loaded);
              } catch (e) {
                setWarmed({ error: -1 });
              } finally {
                setWarming(false);
              }
            }}
          >
            {warming ? (
              <>
                <Loader2 className="size-4 animate-spin" /> loading weights…
              </>
            ) : warmed ? (
              "All models loaded ✓"
            ) : (
              "Load all model weights"
            )}
          </Button>
          {warmed && !warming && (
            <div className="space-y-0.5 text-xs text-muted-foreground">
              {Object.entries(warmed).map(([m, s]) => (
                <div key={m}>
                  {m}: {s < 0 ? "failed" : `${s}s`}
                </div>
              ))}
            </div>
          )}
        </div>
        <div className="space-y-1.5">
          <label className="text-sm font-medium">Model</label>
          <Select value={model} onChange={(e) => setModel(e.target.value)}>
            {models.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </Select>
          <p className="text-xs text-muted-foreground">
            First message per model loads it — later ones show true latency.
          </p>
        </div>
        {(model.startsWith("groq") || model.startsWith("gemini")) && (
          <div className="space-y-1.5">
            <label className="text-sm font-medium">API key (BYOK)</label>
            {model.startsWith("groq") ? (
              <Input
                type="password"
                placeholder="gsk_…"
                value={groqKey}
                onChange={(e) => {
                  setGroqKey(e.target.value);
                  localStorage.setItem("groq_key", e.target.value);
                }}
              />
            ) : (
              <Input
                type="password"
                placeholder="AIza…"
                value={geminiKey}
                onChange={(e) => {
                  setGeminiKey(e.target.value);
                  localStorage.setItem("gemini_key", e.target.value);
                }}
              />
            )}
            <p className="text-xs text-muted-foreground">
              Stays in your browser; sent only to your local backend per request. Empty = server env
              var fallback.
            </p>
            <Button
              variant="secondary"
              className="w-full"
              disabled={fetching}
              onClick={fetchProviderModels}
            >
              {fetching ? "Checking server…" : "Fetch live models"}
            </Button>
            {providerModels.length > 0 && (
              <Select value={providerModel} onChange={(e) => setProviderModel(e.target.value)}>
                <option value="">default ({provider === "groq" ? "llama-3.1-8b-instant" : "gemini-2.5-flash-lite"})</option>
                {providerModels.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </Select>
            )}
            {provErr && <p className="text-xs text-destructive">{provErr}</p>}
          </div>
        )}
        <div className="space-y-1.5 overflow-y-auto">
          <label className="text-sm font-medium">Try one</label>
          {demos.map((d) => (
            <Button
              key={d}
              variant="secondary"
              className="h-auto w-full justify-start whitespace-normal py-2 text-left text-xs font-normal"
              onClick={() => route(d)}
            >
              {d}
            </Button>
          ))}
        </div>
      </aside>

      <main className="flex flex-1 flex-col">
        <div className="flex-1 space-y-4 overflow-y-auto p-6">
          {history.length === 0 && !busy && (
            <p className="mt-16 text-center text-sm text-muted-foreground">
              Type a chat message or pick an example — see which workflow it routes to.
            </p>
          )}
          {history.map((h, i) => (
            <div key={i} className="space-y-2">
              <div className="flex justify-end">
                <div className="max-w-[70%] rounded-xl bg-primary px-4 py-2 text-sm text-primary-foreground">
                  {h.message}
                </div>
              </div>
              <div className="flex justify-start">
                <Card className="max-w-[70%] space-y-1.5 px-4 py-3">
                  {h.error ? (
                    <Badge variant="destructive" className="whitespace-normal text-left">{String(h.error)}</Badge>
                  ) : (
                    <>
                      <div className="flex flex-wrap items-center gap-2">
                        <Badge>{h.intent}</Badge>
                        <span className="text-xs text-muted-foreground">
                          {h.ms?.toFixed(1)} ms
                          {h.loadS && h.loadS > 1 ? ` · +${h.loadS.toFixed(1)}s load` : ""} · {h.model}
                        </span>
                      </div>
                      <p className="text-xs text-muted-foreground">{h.response}</p>
                    </>
                  )}
                </Card>
              </div>
            </div>
          ))}
          {busy && (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="size-4 animate-spin" /> routing…
            </div>
          )}
          <div ref={bottom} />
        </div>
        <form
          className="flex gap-2 border-t border-border p-4"
          onSubmit={(e) => {
            e.preventDefault();
            route(input);
          }}
        >
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="my payment failed twice and I got charged anyway"
            autoFocus
          />
          <Button type="submit" disabled={busy || !input.trim()}>
            <Send className="size-4" />
          </Button>
        </form>
      </main>
    </div>
  );
}
