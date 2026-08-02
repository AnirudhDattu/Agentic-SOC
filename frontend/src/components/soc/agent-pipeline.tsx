import { AnimatePresence, motion } from "motion/react";
import { useEffect, useRef, useState } from "react";
import { Panel } from "./panel";
import { cn } from "@/lib/utils";
import type { Incident } from "@/lib/soc";

const STAGES = [
  { key: "source", label: "UNSW-NB15 dataset", detail: "82,333 labeled flows" },
  { key: "stream", label: "Log streamer", detail: "samples 1 row / 2s" },
  { key: "detect", label: "Detection Agent", detail: "Random Forest · Attack / Benign" },
  { key: "reason", label: "Reasoning Agent", detail: "CrewAI · Groq Llama-3.3-70B" },
  { key: "respond", label: "Response Agent", detail: "action taken · local SQLite" },
] as const;

type LogEntry = {
  id: string;
  status: string;
  method: "llm" | "rules";
  text: string;
};

function describe(incident: Incident): LogEntry {
  const method = incident.reasoning_method === "llm" ? "llm" : "rules";
  const text =
    incident.status === "Attack"
      ? `${incident.attack_type} / ${incident.severity} via ${
          method === "llm" ? "Groq LLM + tool lookup" : "ground-truth rule match"
        } → ${incident.action_taken}`
      : "Classified Benign — no action";
  return { id: incident.id, status: incident.status, method, text };
}

export function AgentPipeline({ latest }: { latest: Incident[] }) {
  const [pulses, setPulses] = useState<{ id: string; attack: boolean }[]>([]);
  const [log, setLog] = useState<LogEntry[]>([]);
  const seen = useRef<Set<string>>(new Set());
  const isFirst = useRef(true);

  useEffect(() => {
    if (isFirst.current) {
      for (const incident of latest) seen.current.add(incident.id);
      setLog(latest.slice(0, 6).map(describe));
      isFirst.current = false;
      return;
    }

    const fresh = latest.filter((i) => !seen.current.has(i.id));
    if (!fresh.length) return;
    for (const incident of fresh) seen.current.add(incident.id);

    setPulses((p) => [
      ...p,
      ...fresh.map((i) => ({ id: i.id, attack: i.status === "Attack" })),
    ]);
    for (const incident of fresh) {
      const id = incident.id;
      window.setTimeout(() => setPulses((p) => p.filter((x) => x.id !== id)), 2100);
    }
    setLog((l) => [...fresh.map(describe).reverse(), ...l].slice(0, 6));
  }, [latest]);

  return (
    <Panel title="Agent pipeline" subtitle="live · dataset → detection → reasoning → response">
      <div className="relative py-5">
        <div className="absolute top-[26px] right-[6%] left-[6%] h-px bg-border" aria-hidden />
        <div className="relative flex justify-between">
          {STAGES.map((stage, i) => (
            <div key={stage.key} className="flex w-[19%] flex-col items-center gap-1.5 text-center">
              <div className="relative z-10 flex size-9 items-center justify-center rounded-full border border-border-strong bg-surface font-mono text-[10px] text-muted-foreground">
                {i + 1}
              </div>
              <p className="text-[11px] leading-tight font-semibold">{stage.label}</p>
              <p className="text-[10px] leading-tight text-muted-foreground">{stage.detail}</p>
            </div>
          ))}
        </div>
        <AnimatePresence>
          {pulses.map((p) => (
            <motion.span
              key={p.id}
              className={cn(
                "absolute top-[22px] size-2 -translate-y-1/2 rounded-full shadow-[0_0_6px_currentColor]",
                p.attack ? "bg-sev-high text-sev-high" : "bg-sev-low text-sev-low",
              )}
              initial={{ left: "6%", opacity: 0 }}
              animate={{ left: "94%", opacity: [0, 1, 1, 0] }}
              transition={{ duration: 2, ease: "linear" }}
            />
          ))}
        </AnimatePresence>
      </div>

      <div className="mt-1 space-y-1 border-t border-border/60 pt-2">
        {log.length === 0 ? (
          <p className="text-[11px] text-muted-foreground">Waiting for the next event…</p>
        ) : (
          log.map((entry) => (
            <div key={entry.id} className="flex items-center gap-2 font-mono text-[11px]">
              <span
                className={cn(
                  "size-1.5 shrink-0 rounded-full",
                  entry.status === "Attack" ? "bg-sev-high" : "bg-sev-low",
                )}
              />
              <span
                className={cn(
                  "shrink-0 rounded-sm border px-1 text-[9px] uppercase",
                  entry.method === "llm"
                    ? "border-primary/50 text-primary"
                    : "border-border-strong text-muted-foreground",
                )}
              >
                {entry.method}
              </span>
              <span className="truncate text-muted-foreground">{entry.text}</span>
            </div>
          ))
        )}
      </div>
    </Panel>
  );
}
