import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { toast } from "sonner";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { SeverityBadge, StatusBadge, WorkflowBadge } from "./badges";
import { updateIncident } from "@/lib/incidents";
import {
  FEATURE_FIELDS,
  WORKFLOW_STATUSES,
  formatDateTime,
  formatNumber,
  type Incident,
} from "@/lib/soc";
import { cn } from "@/lib/utils";

function Field({ label, value, mono = true }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-3 border-b border-border/60 py-1.5">
      <span className="text-[11px] text-muted-foreground">{label}</span>
      <span className={cn("text-right text-xs tabular", mono && "font-mono")}>{value}</span>
    </div>
  );
}

export function IncidentDrawer({
  incident,
  onOpenChange,
}: {
  incident: Incident | null;
  onOpenChange: (open: boolean) => void;
}) {
  const queryClient = useQueryClient();
  const [note, setNote] = useState("");

  useEffect(() => {
    setNote(incident?.analyst_note ?? "");
  }, [incident?.id, incident?.analyst_note]);

  const mutation = useMutation({
    mutationFn: (patch: { workflow_status?: string; analyst_note?: string | null }) =>
      updateIncident(incident!.id, patch),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["incidents"] });
      toast.success("Incident updated");
    },
    onError: (e: Error) => toast.error("Update failed", { description: e.message }),
  });

  const open = Boolean(incident);

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="w-full overflow-y-auto border-border bg-surface p-0 sm:max-w-xl">
        {incident && (
          <>
            <SheetHeader className="border-b border-border p-4">
              <SheetTitle className="flex flex-wrap items-center gap-2 font-mono text-sm tracking-wide">
                {incident.attack_type}
                <SeverityBadge severity={incident.severity} />
                <StatusBadge status={incident.status} />
                <WorkflowBadge status={incident.workflow_status} />
              </SheetTitle>
              <SheetDescription className="font-mono text-xs">
                {formatDateTime(incident.timestamp)} · {incident.id.slice(0, 8)}
              </SheetDescription>
            </SheetHeader>

            <div className="space-y-5 p-4">
              <section>
                <h3 className="mb-2 font-mono text-[10px] tracking-[0.14em] text-muted-foreground uppercase">
                  Analyst workflow
                </h3>
                <div className="flex flex-wrap gap-1.5">
                  {WORKFLOW_STATUSES.map((s) => (
                    <button
                      key={s}
                      type="button"
                      disabled={mutation.isPending}
                      onClick={() => mutation.mutate({ workflow_status: s })}
                      className={cn(
                        "rounded-sm border px-2.5 py-1 font-mono text-[11px] tracking-wide uppercase transition-colors disabled:opacity-50",
                        incident.workflow_status === s
                          ? "border-primary bg-accent text-accent-foreground"
                          : "border-border-strong text-muted-foreground hover:bg-secondary",
                      )}
                    >
                      {s}
                    </button>
                  ))}
                </div>
                <textarea
                  value={note}
                  onChange={(e) => setNote(e.target.value)}
                  rows={3}
                  placeholder="Add an analyst note…"
                  className="mt-2 w-full resize-y rounded-sm border border-input bg-background p-2 font-mono text-xs outline-none focus-visible:ring-1 focus-visible:ring-ring"
                />
                <button
                  type="button"
                  disabled={mutation.isPending || note === (incident.analyst_note ?? "")}
                  onClick={() => mutation.mutate({ analyst_note: note || null })}
                  className="mt-1.5 rounded-sm bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground transition-opacity hover:opacity-90 disabled:opacity-40"
                >
                  {mutation.isPending ? "Saving…" : "Save note"}
                </button>
              </section>

              <section>
                <h3 className="mb-2 font-mono text-[10px] tracking-[0.14em] text-muted-foreground uppercase">
                  LLM recommendation
                </h3>
                <p className="rounded-sm border border-border bg-surface-2 p-2.5 text-xs leading-relaxed">
                  {incident.recommendation || "—"}
                </p>
                <div className="mt-2 grid gap-0 sm:grid-cols-2 sm:gap-x-6">
                  <Field label="Action taken" value={incident.action_taken} />
                  <Field label="Source IP" value={incident.source_ip ?? "—"} />
                  <Field label="Source country" value={incident.source_country ?? "—"} mono={false} />
                  <Field
                    label="Coordinates"
                    value={
                      incident.source_lat != null && incident.source_lon != null
                        ? `${incident.source_lat.toFixed(2)}, ${incident.source_lon.toFixed(2)}`
                        : "—"
                    }
                  />
                </div>
              </section>

              <section>
                <h3 className="mb-2 font-mono text-[10px] tracking-[0.14em] text-muted-foreground uppercase">
                  Raw network-flow features ({FEATURE_FIELDS.length})
                </h3>
                <div className="grid gap-x-6 sm:grid-cols-2">
                  {FEATURE_FIELDS.map((f) => (
                    <Field
                      key={f.key}
                      label={`${f.label}${f.unit ? ` (${f.unit})` : ""}`}
                      value={formatNumber(incident.raw_features?.[f.key])}
                    />
                  ))}
                </div>
              </section>
            </div>
          </>
        )}
      </SheetContent>
    </Sheet>
  );
}
