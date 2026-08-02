import { useEffect, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { SkeletonBlock } from "./panel";

const axis = {
  stroke: "var(--muted-foreground)",
  fontSize: 10,
  fontFamily: "var(--font-mono)",
};

const tooltipStyle = {
  background: "var(--popover)",
  border: "1px solid var(--border-strong)",
  borderRadius: 4,
  fontSize: 11,
  fontFamily: "var(--font-mono)",
  color: "var(--popover-foreground)",
  padding: "6px 8px",
};

function ClientOnlyChart({ children, height }: { children: React.ReactNode; height: number }) {
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  if (!mounted) return <SkeletonBlock className="w-full" />;
  return (
    <div style={{ height }} className="w-full">
      <ResponsiveContainer width="100%" height="100%">
        {children as React.ReactElement}
      </ResponsiveContainer>
    </div>
  );
}

export function AttackTypeChart({
  data,
  height = 220,
}: {
  data: { type: string; count: number }[];
  height?: number;
}) {
  return (
    <ClientOnlyChart height={height}>
      <BarChart data={data} layout="vertical" margin={{ left: 4, right: 12, top: 4, bottom: 4 }}>
        <XAxis type="number" {...axis} tickLine={false} axisLine={false} />
        <YAxis
          type="category"
          dataKey="type"
          width={104}
          {...axis}
          tickLine={false}
          axisLine={false}
        />
        <Tooltip
          contentStyle={tooltipStyle}
          cursor={{ fill: "color-mix(in oklab, var(--primary) 10%, transparent)" }}
        />
        <Bar dataKey="count" name="Incidents" fill="var(--primary)" radius={[0, 3, 3, 0]} barSize={14} />
      </BarChart>
    </ClientOnlyChart>
  );
}

export function SeverityDonut({
  data,
  height = 220,
}: {
  data: { name: string; value: number; color: string }[];
  height?: number;
}) {
  return (
    <ClientOnlyChart height={height}>
      <PieChart>
        <Pie
          data={data}
          dataKey="value"
          nameKey="name"
          innerRadius="58%"
          outerRadius="88%"
          paddingAngle={2}
          stroke="var(--surface)"
          strokeWidth={2}
        >
          {data.map((d) => (
            <Cell key={d.name} fill={d.color} />
          ))}
        </Pie>
        <Tooltip contentStyle={tooltipStyle} />
      </PieChart>
    </ClientOnlyChart>
  );
}

export function VolumeChart({
  data,
  height = 220,
}: {
  data: { bucket: string; attacks: number; benign: number }[];
  height?: number;
}) {
  return (
    <ClientOnlyChart height={height}>
      <AreaChart data={data} margin={{ left: -18, right: 8, top: 8, bottom: 0 }}>
        <defs>
          <linearGradient id="gradAttacks" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--sev-high)" stopOpacity={0.5} />
            <stop offset="100%" stopColor="var(--sev-high)" stopOpacity={0.03} />
          </linearGradient>
          <linearGradient id="gradBenign" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--primary)" stopOpacity={0.35} />
            <stop offset="100%" stopColor="var(--primary)" stopOpacity={0.02} />
          </linearGradient>
        </defs>
        <XAxis dataKey="bucket" {...axis} tickLine={false} axisLine={false} minTickGap={24} />
        <YAxis {...axis} tickLine={false} axisLine={false} width={40} />
        <Tooltip contentStyle={tooltipStyle} />
        <Area
          type="monotone"
          dataKey="benign"
          name="Benign"
          stroke="var(--primary)"
          fill="url(#gradBenign)"
          strokeWidth={1.5}
        />
        <Area
          type="monotone"
          dataKey="attacks"
          name="Attacks"
          stroke="var(--sev-high)"
          fill="url(#gradAttacks)"
          strokeWidth={1.5}
        />
      </AreaChart>
    </ClientOnlyChart>
  );
}
