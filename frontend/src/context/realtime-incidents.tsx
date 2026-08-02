import { useQueryClient } from "@tanstack/react-query";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { toast } from "sonner";
import { useSocSettings } from "@/context/soc-settings";
import { pushDemoIncident } from "@/lib/demo-feed";
import { fetchRecentIncidents } from "@/lib/incidents";
import { isSeverity, type Incident } from "@/lib/soc";

type Ctx = {
  connected: boolean;
  newIds: Set<string>;
  lastEventAt: number | null;
  demoFeed: boolean;
  setDemoFeed: (on: boolean) => void;
};

const RealtimeContext = createContext<Ctx | null>(null);

// No hosted realtime service (local SQLite only) — poll for new rows on the
// same cadence the Python pipeline emits them (every ~2s) instead.
const POLL_MS = 2000;

export function RealtimeIncidentsProvider({ children }: { children: ReactNode }) {
  const queryClient = useQueryClient();
  const { settings } = useSocSettings();
  const [connected, setConnected] = useState(false);
  const [newIds, setNewIds] = useState<Set<string>>(new Set());
  const [lastEventAt, setLastEventAt] = useState<number | null>(null);
  const [demoFeed, setDemoFeed] = useState(false);
  const alertRef = useRef(settings.alertSeverities);
  alertRef.current = settings.alertSeverities;

  const handleInsert = useCallback(
    (incident: Incident) => {
      queryClient.setQueriesData<Incident[]>({ queryKey: ["incidents", "recent"] }, (prev) =>
        prev ? [incident, ...prev.filter((i) => i.id !== incident.id)] : prev,
      );
      queryClient.invalidateQueries({ queryKey: ["incidents", "page"] });
      setLastEventAt(Date.now());
      setNewIds((prev) => {
        const next = new Set(prev);
        next.add(incident.id);
        return next;
      });
      window.setTimeout(() => {
        setNewIds((prev) => {
          const next = new Set(prev);
          next.delete(incident.id);
          return next;
        });
      }, 2600);

      if (
        incident.status === "Attack" &&
        isSeverity(incident.severity) &&
        alertRef.current.includes(incident.severity)
      ) {
        toast.error(`${incident.severity} severity — ${incident.attack_type}`, {
          description: `${incident.source_ip ?? "unknown source"} · ${incident.action_taken}`,
        });
      }
    },
    [queryClient],
  );

  useEffect(() => {
    if (!settings.liveMode) {
      setConnected(false);
      return;
    }

    let cancelled = false;
    let knownIds = new Set<string>();
    let first = true;

    const poll = async () => {
      try {
        const rows = await fetchRecentIncidents(1);
        if (cancelled) return;
        setConnected(true);
        if (!first) {
          const fresh = rows.filter((r) => !knownIds.has(r.id));
          // Oldest-of-the-new first so toasts/animations land in arrival order.
          for (const incident of [...fresh].reverse()) handleInsert(incident);
        }
        knownIds = new Set(rows.map((r) => r.id));
        first = false;
      } catch {
        if (!cancelled) setConnected(false);
      }
    };

    void poll();
    const id = window.setInterval(poll, POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [settings.liveMode, handleInsert]);

  useEffect(() => {
    if (!demoFeed) return;
    const id = window.setInterval(() => {
      void pushDemoIncident().catch(() => setDemoFeed(false));
    }, 2000);
    return () => window.clearInterval(id);
  }, [demoFeed]);

  const value = useMemo(
    () => ({ connected, newIds, lastEventAt, demoFeed, setDemoFeed }),
    [connected, newIds, lastEventAt, demoFeed],
  );

  return <RealtimeContext.Provider value={value}>{children}</RealtimeContext.Provider>;
}

export function useRealtimeIncidents() {
  const ctx = useContext(RealtimeContext);
  if (!ctx) throw new Error("useRealtimeIncidents must be used inside RealtimeIncidentsProvider");
  return ctx;
}
