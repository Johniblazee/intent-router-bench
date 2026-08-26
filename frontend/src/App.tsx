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
      });
  }, []);

  useEffect(() => bottom.current?.scrollIntoView({ behavior: "smooth" }), [history, busy]);

  async function route(message: string) {
    if (!message.trim() || busy) return;
    setInput("");
    setBusy(true);
    try {
      const r = await fetch("/api/route", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, model }),
      });
      const d = await r.json();
      setHistory((h) => [
        ...h,
        r.ok
          ? { message, intent: d.intent, ms: d.ms, loadS: d.load_s, response: d.response, model }
          : { message, model, error: d.detail ?? r.statusText },
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
          <label className="text-sm font-medium">Model</label>
          <Select value={model} onChange={(e) => setModel(e.target.value)}>
            {models.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </Select>
          <p className="text-xs text-muted-foreground">
            API models need GROQ_API_KEY / GEMINI_API_KEY. First message per model loads it — later
            ones show true latency.
          </p>
        </div>
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
                    <Badge variant="destructive">{h.error}</Badge>
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
