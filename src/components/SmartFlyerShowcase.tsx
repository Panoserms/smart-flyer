/**
 * SmartFlyerShowcase: interactive view of the Smart Flyer pipeline.
 *
 * Reads smart_flyer_data.json (written by pipeline/export_smart_flyer.py) and shows
 * how three CPU stages and one LLM stage prune association rules down to flyer bundles:
 *   1. regex stop-item filter, 2. spaCy head-noun pruner, 3. margin & lift gate,
 *   4. LLM judge (local Gemma, only for the pairs that survived 1 to 3).
 *
 * Self-contained: React + Tailwind CSS + motion/react + lucide-react. Pass `data`
 * directly, or `sources` to fetch one or more JSON files; if nothing loads, an
 * embedded copy of the built-in sample is shown.
 */
import { forwardRef, useCallback, useEffect, useMemo, useRef, useState, type CSSProperties, type ReactNode } from "react";
import { AnimatePresence, LayoutGroup, motion, useReducedMotion, type Variants } from "motion/react";
import { Ban, Brain, Check, Cpu, Eye, Layers, Play, Regex, RotateCcw, Scale, ScanSearch, type LucideIcon } from "lucide-react";

/* ------------------------------------------------------------------ contract */

export type Lang = "de" | "en" | "el";

export interface FlyerItem {
  name: string;
  price_eur: number;
  head_noun: string;
  pos_pattern: string;
}

export interface FlyerRule {
  id: string;
  domain: "supermarket_grocery" | "uci_retail" | string;
  antecedent: FlyerItem;
  consequent: FlyerItem;
  metrics: { support: number; confidence: number; lift: number };
  linguistic_checks: {
    is_packaging_or_stop_item: boolean;
    shares_head_noun: boolean;
    passes_margin_gate: boolean;
    status: "approved_bundle" | "rejected_packaging_noise" | "rejected_same_head_noun" | "rejected_low_margin" | "rejected_llm_judge" | string;
    rejection_explanation: string | null;
  };
  promo_mechanics: {
    anchor_item: string;
    discounted_item: string;
    discount_pct: number;
    regular_bundle_price: number;
    promo_bundle_price: number;
    net_basket_lift_eur: number;
  };
  flyer_copy: ({ badge_text: string } & Record<Lang, { title: string; Copy: string }>) | null;
}

export interface SmartFlyerData {
  telemetry: {
    transactions_analyzed: number;
    raw_association_rules: number;
    rejected_packaging_noise: number;
    rejected_same_head_noun: number;
    rejected_low_margin: number;
    rejected_llm_judge: number;
    surviving_complementary_bundles: number;
    llm_calls_saved_pct: number;
    cpu_pruning_time_ms: number;
  };
  rules: FlyerRule[];
}

export interface DataSource {
  label: string;
  url: string;
}

export interface SmartFlyerShowcaseProps {
  /** Render this payload directly (no fetching). */
  data?: SmartFlyerData;
  /** JSON files to fetch; each one that loads becomes a dataset option. */
  sources?: DataSource[];
}

/* ------------------------------------------------------------------ tokens */

const C = {
  bg: "#0d1420",
  panel: "#131c2b",
  raised: "#192436",
  line: "#25324a",
  text: "#e7ebf2",
  muted: "#8d99ad",
  faint: "#5d6a80",
  tag: "#ffd60a", // shelf-tag yellow
  price: "#e5383b", // price red
  ok: "#34d399", // complementary heads
  clash: "#fb7185", // shared head noun
  regex: "#a78bfa",
  margin: "#fbbf24",
  judge: "#38bdf8", // LLM judge
};

const SANS: CSSProperties = { fontFamily: '"Archivo", ui-sans-serif, system-ui, sans-serif' };
const WIDE: CSSProperties = { ...SANS, fontStretch: "118%" };
const NARROW: CSSProperties = { ...SANS, fontStretch: "75%" };
const MONO: CSSProperties = { fontFamily: '"IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace' };

const SPRING = { type: "spring", stiffness: 250, damping: 22 } as const;

type StageKey = "regex" | "head" | "margin" | "judge";
type Filters = Record<StageKey, boolean>;

const STAGES: { key: StageKey; n: number; label: string; detail: string; icon: LucideIcon; color: string }[] = [
  { key: "regex", n: 1, label: "Regex stop-item filter", detail: "Removes carrier bags, postage and Pfand / deposit lines", icon: Regex, color: C.regex },
  { key: "head", n: 2, label: "spaCy head-noun pruner", detail: "Removes variants that share a head noun or compound stem", icon: ScanSearch, color: C.clash },
  { key: "margin", n: 3, label: "Margin & lift gate", detail: "Removes pairs that barely grow the basket or have low lift", icon: Scale, color: C.margin },
  { key: "judge", n: 4, label: "LLM judge", detail: "Local Gemma reads each survivor and keeps only true complements", icon: Brain, color: C.judge },
];
const STAGE = Object.fromEntries(STAGES.map((s) => [s.key, s])) as Record<StageKey, (typeof STAGES)[number]>;

const NONE: Filters = { regex: false, head: false, margin: false, judge: false };
const ALL: Filters = { regex: true, head: true, margin: true, judge: true };

const LANGS: { key: Lang; label: string; flag: string; locale: string }[] = [
  { key: "de", label: "DE", flag: "🇩🇪", locale: "de-DE" },
  { key: "en", label: "EN", flag: "🇬🇧", locale: "en-GB" },
  { key: "el", label: "EL", flag: "🇬🇷", locale: "el-GR" },
];

const FLYER_TEXT: Record<Lang, {
  anchor: string; pull: string; regular: string; bundle: string; expansion: string;
  badge: (p: number) => string; terms: (p: number) => string;
}> = {
  de: {
    anchor: "Zum Vollpreis", pull: "Kombi-Artikel", regular: "Regulär", bundle: "Kombi-Preis",
    expansion: "Mehr im Warenkorb", badge: (p) => `-${p}% KOMBI-RABATT`,
    terms: (p) => `Beim Kauf beider Artikel: ${p} % Rabatt auf den günstigeren Artikel.`,
  },
  en: {
    anchor: "Full price", pull: "Combo item", regular: "Regular", bundle: "Bundle price",
    expansion: "Net basket expansion", badge: (p) => `-${p}% COMBO DISCOUNT`,
    terms: (p) => `Buy both items and the cheaper one is ${p}% off.`,
  },
  el: {
    anchor: "Πλήρης τιμή", pull: "Προϊόν συνδυασμού", regular: "Κανονική τιμή", bundle: "Τιμή πακέτου",
    expansion: "Αύξηση καλαθιού", badge: (p) => `-${p}% ΕΚΠΤΩΣΗ ΣΥΝΔΥΑΣΜΟΥ`,
    terms: (p) => `Με την αγορά και των δύο, ${p}% έκπτωση στο φθηνότερο προϊόν.`,
  },
};

const DEFAULT_SOURCES: DataSource[] = [
  { label: "UCI Online Retail", url: "smart_flyer_data.json" },
  { label: "Built-in sample", url: "smart_flyer_sample.json" },
];

/* ------------------------------------------------------------------ helpers */

function rejectedBy(rule: FlyerRule, f: Filters): StageKey | null {
  const c = rule.linguistic_checks;
  if (f.regex && c.is_packaging_or_stop_item) return "regex";
  if (f.head && c.shares_head_noun) return "head";
  if (f.margin && !c.passes_margin_gate) return "margin";
  // The judge only sees pairs that passed the three rule stages, so its status is exact.
  if (f.judge && c.status === "rejected_llm_judge") return "judge";
  return null;
}

const STATUS_STAGE: Record<string, StageKey> = {
  rejected_packaging_noise: "regex",
  rejected_same_head_noun: "head",
  rejected_low_margin: "margin",
  rejected_llm_judge: "judge",
};

function explain(rule: FlyerRule, stage: StageKey): string {
  // The pipeline stores the explanation of the first stage that failed; other
  // stages get a generic reason.
  if (STATUS_STAGE[rule.linguistic_checks.status] === stage && rule.linguistic_checks.rejection_explanation) {
    return rule.linguistic_checks.rejection_explanation;
  }
  if (stage === "regex") return "Packaging or operational line, not a product";
  if (stage === "head") return "Substitute Variant: Shared syntactic/compound head";
  if (stage === "judge") return "LLM judge: not a complement";
  return "Fails the margin & lift gate";
}

/** Receipt lines arrive in ALL CAPS; show them in title case. */
function pretty(name: string): string {
  if (name !== name.toUpperCase()) return name;
  return name.toLowerCase().replace(/(^|[\s/(-])(\p{L})/gu, (_, sep: string, ch: string) => sep + ch.toUpperCase());
}

function commonSuffix(a: string, b: string): string {
  let n = 0;
  while (n < a.length && n < b.length && a[a.length - 1 - n] === b[b.length - 1 - n]) n++;
  return a.slice(a.length - n);
}

function headRelation(rule: FlyerRule): { equal: boolean; label: string } {
  const a = rule.antecedent.head_noun;
  const b = rule.consequent.head_noun;
  if (!rule.linguistic_checks.shares_head_noun) return { equal: false, label: "Head" };
  if (a.toLowerCase() === b.toLowerCase()) {
    const compound = /Compound Head: ([^)]+)\)/.exec(rule.antecedent.pos_pattern)?.[1];
    return { equal: true, label: compound && compound.includes(" ") ? `Compound "${compound}"` : "Head" };
  }
  const suf = commonSuffix(a.toLowerCase(), b.toLowerCase());
  return { equal: true, label: suf.length >= 3 ? `Compound "-${suf}"` : "Head" };
}

function money(value: number, lang: Lang): string {
  const locale = LANGS.find((l) => l.key === lang)?.locale ?? "en-GB";
  return new Intl.NumberFormat(locale, { style: "currency", currency: "EUR" }).format(value);
}

function useCount(target: number, decimals = 0): string {
  const reduce = useReducedMotion();
  const [value, setValue] = useState(target);
  const from = useRef(target);
  useEffect(() => {
    if (reduce) {
      setValue(target);
      return;
    }
    const start = performance.now();
    const origin = from.current;
    let raf = 0;
    const tick = (t: number) => {
      const k = Math.min(1, (t - start) / 700);
      const eased = 1 - Math.pow(1 - k, 3);
      setValue(origin + (target - origin) * eased);
      if (k < 1) raf = requestAnimationFrame(tick);
      else from.current = target;
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, reduce]);
  return value.toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
}

/* ------------------------------------------------------------------ component */

export default function SmartFlyerShowcase({ data, sources = DEFAULT_SOURCES }: SmartFlyerShowcaseProps) {
  const [datasets, setDatasets] = useState<{ label: string; data: SmartFlyerData }[] | null>(
    data ? [{ label: "Pipeline output", data }] : null,
  );
  const [active, setActive] = useState(0);
  const [lang, setLang] = useState<Lang>("de");
  const [filters, setFilters] = useState<Filters>(NONE);
  const [xray, setXray] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [runStep, setRunStep] = useState(0); // 0 = idle, 1..4 = stage being switched on
  const timers = useRef<number[]>([]);
  const reduce = useReducedMotion();

  useEffect(() => {
    if (data) return;
    let cancelled = false;
    Promise.all(
      sources.map(async (s) => {
        try {
          const res = await fetch(s.url);
          if (!res.ok) return null;
          const json = (await res.json()) as SmartFlyerData;
          return Array.isArray(json.rules) && json.telemetry ? { label: s.label, data: json } : null;
        } catch {
          return null;
        }
      }),
    ).then((loaded) => {
      if (cancelled) return;
      const ok = loaded.filter((d): d is { label: string; data: SmartFlyerData } => d !== null);
      setDatasets(ok.length ? ok : [{ label: "Built-in sample (embedded)", data: EMBEDDED_SAMPLE }]);
    });
    return () => {
      cancelled = true;
    };
  }, [data, sources]);

  useEffect(() => () => timers.current.forEach(clearTimeout), []);

  const current = datasets?.[active]?.data;
  const rules = useMemo(
    () => [...(current?.rules ?? [])].sort((a, b) => b.metrics.lift - a.metrics.lift),
    [current],
  );
  const visible = useMemo(() => rules.filter((r) => xray || !rejectedBy(r, filters)), [rules, filters, xray]);
  const removed = rules.length - rules.filter((r) => !rejectedBy(r, filters)).length;

  const approved = rules.filter((r) => r.linguistic_checks.status === "approved_bundle");
  const selected =
    rules.find((r) => r.id === selectedId) ?? approved[0] ?? rules[0] ?? null;

  // Keep the flyer on a rule that is still on the canvas.
  useEffect(() => {
    if (!xray && selectedId) {
      const r = rules.find((x) => x.id === selectedId);
      if (r && rejectedBy(r, filters)) setSelectedId(null);
    }
  }, [filters, xray, rules, selectedId]);

  const runPruner = useCallback(() => {
    timers.current.forEach(clearTimeout);
    timers.current = [];
    setFilters(NONE);
    setRunStep(1);
    const gap = reduce ? 150 : 900;
    STAGES.forEach((s, i) => {
      timers.current.push(
        window.setTimeout(() => {
          setRunStep(i + 1);
          setFilters((f) => ({ ...f, [s.key]: true }));
        }, gap * (i + 0.5)),
      );
    });
    timers.current.push(window.setTimeout(() => setRunStep(0), gap * (STAGES.length + 0.4)));
  }, [reduce]);

  const reset = () => {
    timers.current.forEach(clearTimeout);
    setRunStep(0);
    setFilters(NONE);
  };

  return (
    <div className="min-h-screen w-full overflow-x-hidden" style={{ ...SANS, background: C.bg, color: C.text }}>
      <div className="mx-auto max-w-[1360px] px-4 pb-16 pt-8 sm:px-6 lg:px-8">
        <Header
          lang={lang}
          setLang={setLang}
          datasets={datasets?.map((d) => d.label) ?? []}
          active={active}
          setActive={(i) => {
            setActive(i);
            setSelectedId(null);
          }}
        />

        {!current ? (
          <p className="mt-16 text-sm" style={{ color: C.muted }}>Loading association rules…</p>
        ) : (
          <>
            <Telemetry data={current} filters={filters} />

            <Controls
              filters={filters}
              toggle={(k) => {
                setRunStep(0);
                setFilters((f) => ({ ...f, [k]: !f[k] }));
              }}
              runStep={runStep}
              onRun={runPruner}
              onReset={reset}
              xray={xray}
              setXray={setXray}
            />

            <p className="mt-6 text-sm" style={{ color: C.muted }}>
              Showing {visible.length} of the {rules.length} rules exported for this view
              {current.telemetry.raw_association_rules > rules.length
                ? ` (the run scored ${current.telemetry.raw_association_rules.toLocaleString("en-US")}: the top bundles plus a sample of each rejection type)`
                : ""}
              .{" "}
              {removed > 0 ? `${removed} ${xray ? "are marked as blocked" : "removed"} by the active stages.` : "Switch a stage on to start pruning."}
            </p>

            <div className="mt-4 grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_400px] xl:grid-cols-[minmax(0,1fr)_440px]">
              <section aria-label="Association rules" className="order-2 lg:order-1">
                <LayoutGroup>
                  <motion.div layout className="grid grid-cols-1 gap-3 md:grid-cols-2">
                    <AnimatePresence mode="popLayout" custom={filters}>
                      {visible.map((rule) => (
                        <RuleCard
                          key={rule.id}
                          rule={rule}
                          filters={filters}
                          xray={xray}
                          selected={selected?.id === rule.id}
                          onSelect={() => setSelectedId(rule.id)}
                          lang={lang}
                        />
                      ))}
                    </AnimatePresence>
                  </motion.div>
                </LayoutGroup>
                {visible.length === 0 && (
                  <p className="rounded-xl border p-6 text-sm" style={{ borderColor: C.line, color: C.muted }}>
                    No rule passes these stages. Switch a stage off to bring rules back.
                  </p>
                )}
              </section>

              <aside className="order-1 lg:order-2" aria-label="Flyer preview">
                <div className="lg:sticky lg:top-6">
                  {selected && <FlyerPanel rule={selected} lang={lang} filters={filters} />}
                  <SourceNote data={current} />
                </div>
              </aside>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ header */

function Header(props: {
  lang: Lang;
  setLang: (l: Lang) => void;
  datasets: string[];
  active: number;
  setActive: (i: number) => void;
}) {
  return (
    <header className="flex flex-col gap-5 md:flex-row md:items-end md:justify-between">
      <div className="max-w-2xl">
        <h1 className="text-[2.6rem] leading-[0.95] font-[850] tracking-[-0.02em] sm:text-6xl" style={NARROW}>
          Smart Flyer
        </h1>
        <p className="mt-3 text-[15px] leading-relaxed" style={{ color: C.muted }}>
          Market basket rules, pruned in three CPU stages before a local language model judges the survivors and writes
          guarded flyer copy.
        </p>
      </div>
      <div className="flex flex-wrap items-center gap-3">
        {props.datasets.length > 1 && (
          <Segmented
            label="Dataset"
            options={props.datasets.map((d, i) => ({ key: String(i), node: d }))}
            value={String(props.active)}
            onChange={(k) => props.setActive(Number(k))}
            id="dataset"
          />
        )}
        <Segmented
          label="Flyer language"
          options={LANGS.map((l) => ({
            key: l.key,
            node: (
              <span className="flex items-center gap-1.5">
                {l.label} <span aria-hidden>{l.flag}</span>
              </span>
            ),
          }))}
          value={props.lang}
          onChange={(k) => props.setLang(k as Lang)}
          id="lang"
        />
      </div>
    </header>
  );
}

function Segmented(props: {
  label: string;
  id: string;
  options: { key: string; node: ReactNode }[];
  value: string;
  onChange: (key: string) => void;
}) {
  return (
    <div role="group" aria-label={props.label} className="flex rounded-full border p-1" style={{ borderColor: C.line, background: C.panel }}>
      {props.options.map((o) => {
        const on = o.key === props.value;
        return (
          <button
            key={o.key}
            type="button"
            aria-pressed={on}
            onClick={() => props.onChange(o.key)}
            className="relative rounded-full px-3.5 py-1.5 text-sm font-semibold outline-none focus-visible:ring-2 focus-visible:ring-[#ffd60a]"
            style={{ color: on ? C.bg : C.muted }}
          >
            {on && (
              <motion.span
                layoutId={`seg-${props.id}`}
                className="absolute inset-0 rounded-full"
                style={{ background: C.tag }}
                transition={SPRING}
              />
            )}
            <span className="relative">{o.node}</span>
          </button>
        );
      })}
    </div>
  );
}

/* ------------------------------------------------------------------ telemetry */

function Telemetry({ data, filters }: { data: SmartFlyerData; filters: Filters }) {
  const t = data.telemetry;
  const raw = Math.max(1, t.raw_association_rules);
  const judgeCount = t.rejected_llm_judge ?? 0;
  const anyFilterOn = STAGES.some((s) => filters[s.key]);
  const activeRejected = anyFilterOn
    ? (filters.regex ? t.rejected_packaging_noise : 0) +
      (filters.head ? t.rejected_same_head_noun : 0) +
      (filters.margin ? t.rejected_low_margin : 0) +
      (filters.judge ? judgeCount : 0)
    : t.rejected_packaging_noise + t.rejected_same_head_noun + t.rejected_low_margin + judgeCount;
  const activeKeep = Math.max(0, t.raw_association_rules - activeRejected);
  const cpuSurvivors = Math.max(
    0,
    t.raw_association_rules - t.rejected_packaging_noise - t.rejected_same_head_noun - t.rejected_low_margin,
  );
  const segments: { key: StageKey | "keep"; count: number; color: string; label: string }[] = [
    { key: "regex", count: t.rejected_packaging_noise, color: C.regex, label: "packaging & stop items" },
    { key: "head", count: t.rejected_same_head_noun, color: C.clash, label: "shared head noun" },
    { key: "margin", count: t.rejected_low_margin, color: C.margin, label: "margin & lift gate" },
    { key: "judge", count: judgeCount, color: C.judge, label: "LLM judge" },
    { key: "keep", count: activeKeep, color: C.ok, label: "flyer bundles" },
  ];
  const rawText = useCount(t.raw_association_rules);
  const keepText = useCount(activeKeep);
  const msText = useCount(t.cpu_pruning_time_ms, 1);
  const savedText = useCount(t.llm_calls_saved_pct, 1);

  return (
    <section aria-label="Pipeline telemetry" className="mt-8 rounded-2xl border p-5 sm:p-6" style={{ borderColor: C.line, background: C.panel }}>
      <p className="text-2xl leading-snug font-[650] sm:text-[1.9rem]" style={NARROW}>
        <span className="tabular-nums font-[850]">{rawText}</span> association rules{" "}
        <span style={{ color: C.faint }}>→</span>{" "}
        <span className="tabular-nums font-[850]" style={{ color: C.ok }}>{keepText}</span> flyer bundles in{" "}
        <span className="tabular-nums font-[850]">{msText} ms</span> on CPU.{" "}
        <span style={{ color: C.muted }}>
          <span className="tabular-nums font-[850]" style={{ color: C.text }}>{savedText}%</span> fewer LLM calls.
        </span>
      </p>

      <div className="mt-5 flex h-4 w-full overflow-hidden rounded-full" style={{ background: C.raised }} role="img"
        aria-label={segments.map((s) => `${s.count} ${s.label}`).join(", ")}>
        {segments.map((s) => {
          const dim = s.key !== "keep" && !filters[s.key];
          return (
            <motion.div
              key={s.key}
              className="h-full"
              initial={false}
              animate={{ width: `${(s.count / raw) * 100}%`, opacity: dim ? 0.28 : 1 }}
              transition={SPRING}
              style={{ background: s.color }}
            />
          );
        })}
      </div>

      <dl className="mt-4 grid grid-cols-2 gap-x-6 gap-y-3 text-sm sm:grid-cols-5">
        {segments.map((s) => (
          <div key={s.key} className="flex items-baseline gap-2">
            <span className="mt-1 inline-block h-2.5 w-2.5 shrink-0 rounded-sm" style={{ background: s.color }} />
            <dt style={{ color: C.muted }}>{s.label}</dt>
            <dd className="ml-auto tabular-nums font-semibold">{s.count.toLocaleString("en-US")}</dd>
          </div>
        ))}
      </dl>

      <div className="mt-4 flex flex-wrap gap-x-6 gap-y-1 text-xs" style={{ color: C.faint }}>
        <span className="flex items-center gap-1.5">
          <Layers size={13} aria-hidden />
          {t.transactions_analyzed > 0
            ? `${t.transactions_analyzed.toLocaleString("en-US")} transactions (invoices) mined`
            : "Built-in sample: rule metrics are illustrative"}
        </span>
        <span className="flex items-center gap-1.5">
          <Cpu size={13} aria-hidden />
          {judgeCount > 0
            ? `CPU rules keep ${cpuSurvivors} in ${t.cpu_pruning_time_ms.toFixed(1)} ms; LLM judge removes ${judgeCount} → ${t.surviving_complementary_bundles} bundles`
            : "Timing covers parsing and the three rule stages, not the LLM judge or model loading"}
        </span>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ controls */

function Controls(props: {
  filters: Filters;
  toggle: (k: StageKey) => void;
  runStep: number;
  onRun: () => void;
  onReset: () => void;
  xray: boolean;
  setXray: (v: boolean) => void;
}) {
  const allOn = STAGES.every((s) => props.filters[s.key]);
  return (
    <section aria-label="Pruner controls" className="mt-6">
      <ol className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
        {STAGES.map((s) => {
          const on = props.filters[s.key];
          const running = props.runStep === s.n;
          const Icon = s.icon;
          return (
            <li key={s.key}>
              <button
                type="button"
                role="switch"
                aria-checked={on}
                onClick={() => props.toggle(s.key)}
                className="group flex h-full w-full items-start gap-3 rounded-xl border p-4 text-left outline-none transition-colors focus-visible:ring-2 focus-visible:ring-[#ffd60a]"
                style={{ borderColor: on ? s.color : C.line, background: on ? `${s.color}14` : C.panel }}
              >
                <span
                  className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-sm font-bold tabular-nums"
                  style={{ background: on ? s.color : C.raised, color: on ? C.bg : C.muted }}
                >
                  {s.n}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="flex items-center gap-2 font-semibold">
                    <Icon size={15} aria-hidden style={{ color: s.color }} /> {s.label}
                  </span>
                  <span className="mt-1 block text-[13px] leading-snug" style={{ color: C.muted }}>{s.detail}</span>
                </span>
                <span
                  aria-hidden
                  className="relative mt-1 h-6 w-11 shrink-0 rounded-full transition-colors"
                  style={{ background: on ? s.color : C.line }}
                >
                  <motion.span
                    className="absolute top-1 h-4 w-4 rounded-full"
                    style={{ background: on ? C.bg : C.muted }}
                    animate={{ left: on ? 24 : 4, scale: running ? [1, 1.35, 1] : 1 }}
                    transition={SPRING}
                  />
                </span>
              </button>
            </li>
          );
        })}
      </ol>

      <div className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={props.onRun}
            disabled={props.runStep > 0}
            className="flex items-center gap-2 rounded-full px-5 py-2.5 text-sm font-bold outline-none focus-visible:ring-2 focus-visible:ring-white disabled:opacity-70"
            style={{ background: C.tag, color: C.bg }}
          >
            <Play size={15} aria-hidden fill="currentColor" />
            {props.runStep > 0 ? `Running stage ${props.runStep} of ${STAGES.length}…` : "Run full linguistic pruner"}
          </button>
          <button
            type="button"
            onClick={props.onReset}
            disabled={!STAGES.some((s) => props.filters[s.key])}
            className="flex items-center gap-2 rounded-full border px-4 py-2.5 text-sm font-semibold outline-none focus-visible:ring-2 focus-visible:ring-[#ffd60a] disabled:opacity-40"
            style={{ borderColor: C.line, color: C.text }}
          >
            <RotateCcw size={14} aria-hidden /> Show raw rules
          </button>
          {allOn && props.runStep === 0 && (
            <span className="flex items-center gap-1.5 self-center text-sm" style={{ color: C.ok }}>
              <Check size={15} aria-hidden /> All stages on
            </span>
          )}
        </div>
        <Segmented
          label="View mode"
          id="view"
          value={props.xray ? "xray" : "canvas"}
          onChange={(k) => props.setXray(k === "xray")}
          options={[
            { key: "canvas", node: <span className="flex items-center gap-1.5"><Layers size={14} aria-hidden /> Spatial promo canvas</span> },
            { key: "xray", node: <span className="flex items-center gap-1.5"><Eye size={14} aria-hidden /> Syntactic X-ray</span> },
          ]}
        />
      </div>
      <p className="mt-2 text-xs" style={{ color: C.faint }}>
        {props.xray
          ? "X-ray keeps blocked rules on screen and shows why each one was blocked."
          : "The canvas removes blocked rules, so only flyer candidates remain."}
      </p>
    </section>
  );
}

/* ------------------------------------------------------------------ rule card */

interface RuleCardProps {
  rule: FlyerRule;
  filters: Filters;
  xray: boolean;
  selected: boolean;
  onSelect: () => void;
  lang: Lang;
}

// forwardRef: AnimatePresence mode="popLayout" measures exiting cards through a ref.
const RuleCard = forwardRef<HTMLButtonElement, RuleCardProps>(function RuleCard(props, ref) {
  const { rule, filters, xray, selected } = props;
  const reduce = useReducedMotion();
  const blocked = rejectedBy(rule, filters);
  const rel = headRelation(rule);
  const headOn = filters.head && rule.linguistic_checks.shares_head_noun;

  // Exit animation depends on the filters at removal time, passed via AnimatePresence `custom`.
  const variants: Variants = {
    initial: { opacity: 0, scale: 0.96 },
    enter: { opacity: 1, scale: 1, x: 0, transition: SPRING },
    exit: (f: Filters) => {
      const why = rejectedBy(rule, f ?? filters);
      if (why === "head" && !reduce) {
        return {
          x: [0, -12, 12, -9, 9, -4, 0],
          opacity: [1, 1, 1, 1, 1, 1, 0],
          scale: [1, 1, 1, 1, 1, 0.98, 0.85],
          transition: { duration: 0.8, times: [0, 0.1, 0.2, 0.3, 0.4, 0.55, 1] },
        };
      }
      return { opacity: 0, scale: 0.9, transition: { duration: reduce ? 0.1 : 0.3 } };
    },
  };

  const stage = blocked ? STAGE[blocked] : null;
  const domain = rule.domain === "uci_retail" ? "UCI Online Retail" : rule.domain === "supermarket_grocery" ? "Grocery" : rule.domain;

  return (
    <motion.button
      ref={ref}
      type="button"
      aria-label={`${pretty(rule.antecedent.name)} plus ${pretty(rule.consequent.name)}, lift ${rule.metrics.lift.toFixed(2)}${stage && blocked ? `, blocked at stage ${stage.n}` : ""}`}
      layout
      variants={variants}
      initial="initial"
      animate="enter"
      exit="exit"
      transition={SPRING}
      onClick={props.onSelect}
      aria-pressed={selected}
      className="relative flex w-full flex-col rounded-xl border p-4 text-left outline-none focus-visible:ring-2 focus-visible:ring-[#ffd60a]"
      style={{
        background: selected ? C.raised : C.panel,
        borderColor: selected ? C.tag : stage ? `${stage.color}66` : C.line,
        opacity: stage ? 0.82 : 1,
      }}
    >
      <div className="flex items-center justify-between gap-2 text-xs">
        <span style={{ color: C.faint }}>{domain}</span>
        <span className="rounded-md px-2 py-0.5 font-semibold tabular-nums" style={{ background: C.raised, color: C.text }}>
          Lift {rule.metrics.lift.toFixed(2)}
        </span>
      </div>

      <ItemLine item={rule.antecedent} lang={props.lang} />
      <span aria-hidden className="my-0.5 text-sm font-bold" style={{ color: C.faint }}>+</span>
      <ItemLine item={rule.consequent} lang={props.lang} />

      <motion.div
        key={headOn ? "clash" : "calm"}
        animate={headOn && !reduce ? { x: [0, -5, 5, -3, 3, 0] } : { x: 0 }}
        transition={{ duration: 0.45 }}
        className="mt-3 flex flex-wrap items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs"
        style={{
          ...MONO,
          borderColor: rel.equal ? `${C.clash}80` : `${C.ok}66`,
          background: rel.equal ? `${C.clash}14` : `${C.ok}10`,
          color: rel.equal ? C.clash : C.ok,
        }}
      >
        <span style={{ color: C.muted }}>{rel.label}:</span>
        <span>"{rule.antecedent.head_noun}"</span>
        <span className="font-bold">{rel.equal ? "==" : "≠"}</span>
        <span>"{rule.consequent.head_noun}"</span>
      </motion.div>

      {xray && (
        <div className="mt-2 space-y-0.5 text-[11px] leading-snug" style={{ ...MONO, color: C.faint }}>
          <div className="truncate">A: {rule.antecedent.pos_pattern}</div>
          <div className="truncate">B: {rule.consequent.pos_pattern}</div>
        </div>
      )}

      <div className="mt-3 flex gap-4 text-[11px] tabular-nums" style={{ color: C.faint }}>
        <span>support {(rule.metrics.support * 100).toFixed(1)}%</span>
        <span>confidence {(rule.metrics.confidence * 100).toFixed(0)}%</span>
      </div>

      {stage && blocked && (
        <div
          className="mt-3 flex items-start gap-2 rounded-lg px-2.5 py-2 text-xs leading-snug"
          style={{ background: `${stage.color}1f`, color: stage.color }}
        >
          <Ban size={14} aria-hidden className="mt-px shrink-0" />
          <span>
            <span className="font-semibold">Blocked at stage {stage.n}.</span>{" "}
            <span style={{ color: C.text }}>{explain(rule, blocked)}</span>
          </span>
        </div>
      )}
    </motion.button>
  );
});

function ItemLine({ item, lang }: { item: FlyerItem; lang: Lang }) {
  return (
    <div className="mt-2 flex items-baseline justify-between gap-3">
      <span className="min-w-0 text-[13px] leading-snug" style={MONO}>{item.name}</span>
      <span className="shrink-0 text-sm font-semibold tabular-nums">{money(item.price_eur, lang)}</span>
    </div>
  );
}

/* ------------------------------------------------------------------ flyer */

function PriceTag({ value, lang, size }: { value: number; lang: Lang; size: "lg" | "sm" }) {
  const euros = Math.floor(value + 1e-9);
  const cents = Math.round((value - euros) * 100).toString().padStart(2, "0");
  const sep = lang === "en" ? "." : ",";
  const big = size === "lg";
  return (
    <span className="inline-flex items-start leading-none tabular-nums" style={WIDE} aria-label={money(value, lang)}>
      {lang === "en" && <span className={big ? "mt-1 text-2xl font-[800]" : "text-sm font-[800]"}>€</span>}
      <span className={big ? "text-[3.6rem] font-[900] tracking-[-0.04em]" : "text-xl font-[850]"}>{euros}</span>
      <span className={big ? "mt-1 text-2xl font-[900]" : "text-xs font-[850]"}>
        {lang === "en" ? "." : sep}
        {cents}
      </span>
      {lang !== "en" && <span className={big ? "mt-1 ml-0.5 text-2xl font-[800]" : "ml-0.5 text-sm font-[800]"}>€</span>}
    </span>
  );
}

function FlyerPanel({ rule, lang, filters }: { rule: FlyerRule; lang: Lang; filters: Filters }) {
  const reduce = useReducedMotion();
  const t = FLYER_TEXT[lang];
  const m = rule.promo_mechanics;
  const anchorItem = [rule.antecedent, rule.consequent].find((i) => i.name === m.anchor_item) ?? rule.antecedent;
  const pullItem = [rule.antecedent, rule.consequent].find((i) => i.name === m.discounted_item) ?? rule.consequent;
  const pullPrice = Math.round(pullItem.price_eur * (1 - m.discount_pct / 100) * 100) / 100;
  const blocked = rejectedBy(rule, ALL);
  const copy = rule.flyer_copy?.[lang];

  if (blocked || !rule.flyer_copy || !copy) {
    const stage = STAGE[blocked ?? "margin"];
    return (
      <AnimatePresence mode="wait">
        <motion.div
          key={`blocked-${rule.id}`}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0 }}
          transition={SPRING}
          className="rounded-2xl border p-6"
          style={{ borderColor: `${stage.color}80`, background: C.panel }}
        >
          <p className="flex items-center gap-2 text-sm font-semibold" style={{ color: stage.color }}>
            <Ban size={16} aria-hidden /> Kept off the flyer by stage {stage.n}
          </p>
          <p className="mt-3 text-xl leading-snug font-[750]" style={NARROW}>
            {pretty(rule.antecedent.name)} + {pretty(rule.consequent.name)}
          </p>
          <p className="mt-3 text-sm leading-relaxed" style={{ color: C.muted }}>{explain(rule, stage.key)}</p>
          <dl className="mt-5 grid grid-cols-2 gap-3 text-sm">
            <div><dt style={{ color: C.faint }}>Would-be bundle</dt><dd className="font-semibold tabular-nums">{money(m.promo_bundle_price, lang)}</dd></div>
            <div><dt style={{ color: C.faint }}>Basket expansion</dt><dd className="font-semibold tabular-nums">+{money(m.net_basket_lift_eur, lang)}</dd></div>
          </dl>
          {!filters[stage.key] && (
            <p className="mt-5 text-xs" style={{ color: C.faint }}>Switch on stage {stage.n} to remove this rule from the canvas.</p>
          )}
        </motion.div>
      </AnimatePresence>
    );
  }

  return (
    <AnimatePresence mode="wait">
      <motion.article
        key={`${rule.id}-${lang}`}
        initial={{ opacity: 0, y: 10, rotate: reduce ? 0 : -0.6 }}
        animate={{ opacity: 1, y: 0, rotate: 0 }}
        exit={{ opacity: 0, y: -6 }}
        transition={SPRING}
        className="overflow-hidden rounded-2xl text-[#141414] shadow-[0_24px_60px_-20px_rgba(0,0,0,0.65)]"
        style={{ background: "#ffffff" }}
        lang={lang}
      >
        <div className="relative px-6 pb-5 pt-5" style={{ background: C.tag }}>
          <span className="inline-block rounded-md px-2.5 py-1 text-xs font-[850] tracking-wide text-white" style={{ ...WIDE, background: C.price }}>
            {rule.flyer_copy.badge_text}
          </span>
          <h2 className="mt-3 text-[1.9rem] leading-[1.02] font-[850] tracking-[-0.01em]" style={NARROW}>
            {copy.title}
          </h2>
          <p className="mt-2 text-[15px] leading-snug">{copy.Copy}</p>
        </div>

        <div className="px-6 pt-5">
          <div className="flex items-start justify-between gap-4 border-b border-dashed border-[#d9d9d9] pb-4">
            <div className="min-w-0">
              <p className="text-[11px] font-semibold tracking-wide text-[#6b6b6b]">{t.anchor}</p>
              <p className="mt-1 text-[15px] leading-snug font-semibold">{pretty(anchorItem.name)}</p>
            </div>
            <PriceTag value={anchorItem.price_eur} lang={lang} size="sm" />
          </div>

          <div className="flex items-start justify-between gap-4 pt-4">
            <div className="min-w-0">
              <p className="text-[11px] font-semibold tracking-wide text-[#6b6b6b]">{t.pull}</p>
              <p className="mt-1 text-[15px] leading-snug font-semibold">{pretty(pullItem.name)}</p>
              <motion.span
                className="mt-2 inline-block rounded-md px-2 py-1 text-[11px] font-[850] text-white"
                style={{ ...WIDE, background: C.price }}
                animate={reduce ? undefined : { boxShadow: ["0 0 0px rgba(229,56,59,0.0)", "0 0 18px rgba(229,56,59,0.75)", "0 0 0px rgba(229,56,59,0.0)"] }}
                transition={{ duration: 2.2, repeat: Infinity, ease: "easeInOut" }}
              >
                {t.badge(m.discount_pct)}
              </motion.span>
            </div>
            <div className="flex flex-col items-end">
              <span className="text-sm text-[#8a8a8a] line-through tabular-nums">{money(pullItem.price_eur, lang)}</span>
              <span style={{ color: C.price }}><PriceTag value={pullPrice} lang={lang} size="sm" /></span>
            </div>
          </div>
        </div>

        <div className="mx-6 mt-5 flex items-end justify-between gap-4 rounded-xl px-5 py-4" style={{ background: "#141414", color: "#ffffff" }}>
          <div>
            <p className="text-xs text-[#b5b5b5]">{t.bundle}</p>
            <p className="mt-1 text-xs text-[#b5b5b5]">
              {t.regular} <span className="line-through tabular-nums">{money(m.regular_bundle_price, lang)}</span>
            </p>
          </div>
          <span style={{ color: C.tag }}><PriceTag value={m.promo_bundle_price} lang={lang} size="lg" /></span>
        </div>

        <div className="flex items-center justify-between gap-3 px-6 pb-5 pt-4 text-sm">
          <span className="text-[#555]">{t.expansion}</span>
          <span className="font-[850] tabular-nums" style={{ ...WIDE, color: "#0f7a4f" }}>+{money(m.net_basket_lift_eur, lang)}</span>
        </div>
        <p className="border-t border-[#ececec] px-6 py-3 text-[11px] leading-snug text-[#7a7a7a]">{t.terms(m.discount_pct)}</p>
      </motion.article>
    </AnimatePresence>
  );
}

function SourceNote({ data }: { data: SmartFlyerData }) {
  const real = data.telemetry.transactions_analyzed > 0;
  return (
    <p className="mt-4 text-xs leading-relaxed" style={{ color: C.faint }}>
      {real
        ? "Prices are median UCI unit prices converted from GBP. Flyer copy comes from local Gemma for DE and EN, and from fixed templates when Gemma's output fails validation and for Greek."
        : "Built-in sample: real item pairs, illustrative lift and support. Flyer copy comes from local Gemma for DE and EN, and from fixed templates for Greek."}
    </p>
  );
}

/* ------------------------------------------------------------------ embedded fallback */

// Built-in sample written by `python pipeline/export_smart_flyer.py`; used when no JSON file loads.
const EMBEDDED_SAMPLE: SmartFlyerData = /*EMBED:START*/ {"telemetry": {"transactions_analyzed": 0, "raw_association_rules": 16, "rejected_packaging_noise": 3, "rejected_same_head_noun": 5, "rejected_low_margin": 2, "rejected_llm_judge": 0, "surviving_complementary_bundles": 6, "llm_calls_saved_pct": 6.2, "cpu_pruning_time_ms": 28.6}, "rules": [{"id": "rule-01", "domain": "uci_retail", "antecedent": {"name": "ALARM CLOCK BAKELIKE GREEN", "price_eur": 4.35, "head_noun": "Clock", "pos_pattern": "PROPN (Head: Clock)"}, "consequent": {"name": "ALARM CLOCK BAKELIKE RED", "price_eur": 4.35, "head_noun": "Clock", "pos_pattern": "PROPN (Head: Clock)"}, "metrics": {"support": 0.022, "confidence": 0.64, "lift": 16.3}, "linguistic_checks": {"is_packaging_or_stop_item": false, "shares_head_noun": true, "passes_margin_gate": true, "status": "rejected_same_head_noun", "rejection_explanation": "Substitute Variant: Shared syntactic/compound head"}, "promo_mechanics": {"anchor_item": "ALARM CLOCK BAKELIKE GREEN", "discounted_item": "ALARM CLOCK BAKELIKE RED", "discount_pct": 20, "regular_bundle_price": 8.7, "promo_bundle_price": 7.83, "net_basket_lift_eur": 3.48}, "flyer_copy": null}, {"id": "rule-02", "domain": "uci_retail", "antecedent": {"name": "ROSES REGENCY TEACUP AND SAUCER", "price_eur": 3.42, "head_noun": "Teacup", "pos_pattern": "PROPN (Head: Teacup)"}, "consequent": {"name": "GREEN REGENCY TEACUP AND SAUCER", "price_eur": 3.42, "head_noun": "Teacup", "pos_pattern": "PROPN (Head: Teacup)"}, "metrics": {"support": 0.031, "confidence": 0.71, "lift": 15.8}, "linguistic_checks": {"is_packaging_or_stop_item": false, "shares_head_noun": true, "passes_margin_gate": true, "status": "rejected_same_head_noun", "rejection_explanation": "Substitute Variant: Shared syntactic/compound head"}, "promo_mechanics": {"anchor_item": "ROSES REGENCY TEACUP AND SAUCER", "discounted_item": "GREEN REGENCY TEACUP AND SAUCER", "discount_pct": 20, "regular_bundle_price": 6.84, "promo_bundle_price": 6.16, "net_basket_lift_eur": 2.74}, "flyer_copy": null}, {"id": "rule-03", "domain": "uci_retail", "antecedent": {"name": "WHITE HANGING HEART T-LIGHT HOLDER", "price_eur": 3.42, "head_noun": "Holder", "pos_pattern": "VERB + PROPN + NOUN + PROPN (Head: Holder)"}, "consequent": {"name": "RED HANGING HEART T-LIGHT HOLDER", "price_eur": 3.42, "head_noun": "Holder", "pos_pattern": "VERB + PROPN + NOUN + PROPN (Head: Holder)"}, "metrics": {"support": 0.021, "confidence": 0.52, "lift": 8.9}, "linguistic_checks": {"is_packaging_or_stop_item": false, "shares_head_noun": true, "passes_margin_gate": true, "status": "rejected_same_head_noun", "rejection_explanation": "Substitute Variant: Shared syntactic/compound head"}, "promo_mechanics": {"anchor_item": "WHITE HANGING HEART T-LIGHT HOLDER", "discounted_item": "RED HANGING HEART T-LIGHT HOLDER", "discount_pct": 20, "regular_bundle_price": 6.84, "promo_bundle_price": 6.16, "net_basket_lift_eur": 2.74}, "flyer_copy": null}, {"id": "rule-04", "domain": "supermarket_grocery", "antecedent": {"name": "Grillkohle Buchenholz 3kg", "price_eur": 5.99, "head_noun": "Grillkohle", "pos_pattern": "NOUN + PROPN (Compound Head: -kohle)"}, "consequent": {"name": "Thüringer Rostbratwurst 6 Stück", "price_eur": 3.49, "head_noun": "Rostbratwurst", "pos_pattern": "ADJ + NOUN (Compound Head: -wurst)"}, "metrics": {"support": 0.022, "confidence": 0.51, "lift": 6.8}, "linguistic_checks": {"is_packaging_or_stop_item": false, "shares_head_noun": false, "passes_margin_gate": true, "status": "approved_bundle", "rejection_explanation": null}, "promo_mechanics": {"anchor_item": "Grillkohle Buchenholz 3kg", "discounted_item": "Thüringer Rostbratwurst 6 Stück", "discount_pct": 20, "regular_bundle_price": 9.48, "promo_bundle_price": 8.78, "net_basket_lift_eur": 2.79}, "flyer_copy": {"badge_text": "-20% KOMBI-DEAL", "de": {"title": "Spar-Angebot: Grillkohle & Wurst", "Copy": "Kaufe Grillkohle Buchenholz 3kg und erhalte 20% Rabatt auf die Thüringer Rostbratwurst 6 Stück!"}, "en": {"title": "Great Deal: Charcoal & Sausage", "Copy": "Buy Grillkohle Buchenholz 3kg and get 20% off the Thüringer Rostbratwurst 6 Stück!"}, "el": {"title": "Προσφορά της εβδομάδας", "Copy": "Αγοράστε το προϊόν «Grillkohle Buchenholz 3kg» και κερδίστε 20% έκπτωση στο προϊόν «Thüringer Rostbratwurst 6 Stück»."}}}, {"id": "rule-05", "domain": "uci_retail", "antecedent": {"name": "JAM MAKING SET WITH JARS", "price_eur": 4.93, "head_noun": "Set", "pos_pattern": "PROPN (Head: Set)"}, "consequent": {"name": "RECIPE BOX PANTRY YELLOW DESIGN", "price_eur": 3.42, "head_noun": "Box", "pos_pattern": "PROPN (Head: Box)"}, "metrics": {"support": 0.017, "confidence": 0.36, "lift": 5.4}, "linguistic_checks": {"is_packaging_or_stop_item": false, "shares_head_noun": false, "passes_margin_gate": true, "status": "approved_bundle", "rejection_explanation": null}, "promo_mechanics": {"anchor_item": "JAM MAKING SET WITH JARS", "discounted_item": "RECIPE BOX PANTRY YELLOW DESIGN", "discount_pct": 20, "regular_bundle_price": 8.35, "promo_bundle_price": 7.67, "net_basket_lift_eur": 2.74}, "flyer_copy": {"badge_text": "-20% KOMBI-DEAL", "de": {"title": "Super Spar-Angebot Jetzt!", "Copy": "Kaufe das Jam Making Set With Jars und erhalte die Recipe Box Pantry Yellow Design mit 20% Rabatt!"}, "en": {"title": "Great Deal: Save 20% Now!", "Copy": "Buy the Jam Making Set With Jars and get the Recipe Box Pantry Yellow Design for 20% off!"}, "el": {"title": "Το τέλειο δίδυμο", "Copy": "Αγοράστε το προϊόν «Jam Making Set With Jars» και κερδίστε 20% έκπτωση στο προϊόν «Recipe Box Pantry Yellow Design»."}}}, {"id": "rule-06", "domain": "uci_retail", "antecedent": {"name": "POSTAGE", "price_eur": 20.88, "head_noun": "Postage", "pos_pattern": "NOUN (Head: Postage)"}, "consequent": {"name": "ROUND SNACK BOXES SET OF4 WOODLAND", "price_eur": 3.42, "head_noun": "Box", "pos_pattern": "PROPN (Head: Box)"}, "metrics": {"support": 0.019, "confidence": 0.33, "lift": 5.1}, "linguistic_checks": {"is_packaging_or_stop_item": true, "shares_head_noun": false, "passes_margin_gate": true, "status": "rejected_packaging_noise", "rejection_explanation": "Packaging/operational noise: \"POSTAGE\" in \"Postage\""}, "promo_mechanics": {"anchor_item": "POSTAGE", "discounted_item": "ROUND SNACK BOXES SET OF4 WOODLAND", "discount_pct": 20, "regular_bundle_price": 24.3, "promo_bundle_price": 23.62, "net_basket_lift_eur": 2.74}, "flyer_copy": null}, {"id": "rule-07", "domain": "supermarket_grocery", "antecedent": {"name": "Erdbeerjoghurt 500g", "price_eur": 0.89, "head_noun": "Erdbeerjoghurt", "pos_pattern": "NOUN (Compound Head: -joghurt)"}, "consequent": {"name": "Vanillejoghurt 500g", "price_eur": 0.89, "head_noun": "Vanillejoghurt", "pos_pattern": "NOUN (Compound Head: -joghurt)"}, "metrics": {"support": 0.027, "confidence": 0.44, "lift": 4.6}, "linguistic_checks": {"is_packaging_or_stop_item": false, "shares_head_noun": true, "passes_margin_gate": true, "status": "rejected_same_head_noun", "rejection_explanation": "Substitute Variant: Shared syntactic/compound head"}, "promo_mechanics": {"anchor_item": "Erdbeerjoghurt 500g", "discounted_item": "Vanillejoghurt 500g", "discount_pct": 20, "regular_bundle_price": 1.78, "promo_bundle_price": 1.6, "net_basket_lift_eur": 0.71}, "flyer_copy": null}, {"id": "rule-08", "domain": "supermarket_grocery", "antecedent": {"name": "Bio Espressobohnen Ganze Bohne 1kg", "price_eur": 12.99, "head_noun": "Espressobohne", "pos_pattern": "NOUN + ADJ + NOUN (Compound Head: -bohne)"}, "consequent": {"name": "Barista Haferdrink Ungesüßt 1L", "price_eur": 2.19, "head_noun": "Haferdrink", "pos_pattern": "NOUN (Compound Head: -drink)"}, "metrics": {"support": 0.034, "confidence": 0.48, "lift": 4.15}, "linguistic_checks": {"is_packaging_or_stop_item": false, "shares_head_noun": false, "passes_margin_gate": true, "status": "approved_bundle", "rejection_explanation": null}, "promo_mechanics": {"anchor_item": "Bio Espressobohnen Ganze Bohne 1kg", "discounted_item": "Barista Haferdrink Ungesüßt 1L", "discount_pct": 20, "regular_bundle_price": 15.18, "promo_bundle_price": 14.74, "net_basket_lift_eur": 1.75}, "flyer_copy": {"badge_text": "-20% KOMBI-DEAL", "de": {"title": "Spar-Angebot: Espresso & Haferdrink", "Copy": "Kaufe Bio Espressobohnen Ganze Bohne 1kg und erhalte den Barista Haferdrink Ungesüßt 1L mit 20% Rabatt!"}, "en": {"title": "Espresso and Oat Milk Deal", "Copy": "Get 20% off the Barista Haferdrink Ungesüßt 1L when you buy the Bio Espressobohnen Ganze Bohne 1kg."}, "el": {"title": "Μαζί πιο οικονομικά", "Copy": "Αγοράστε το προϊόν «Bio Espressobohnen Ganze Bohne 1kg» και κερδίστε 20% έκπτωση στο προϊόν «Barista Haferdrink Ungesüßt 1L»."}}}, {"id": "rule-09", "domain": "uci_retail", "antecedent": {"name": "REGENCY CAKESTAND 3 TIER", "price_eur": 14.79, "head_noun": "Cakestand", "pos_pattern": "NOUN (Head: Cakestand)"}, "consequent": {"name": "ROSES REGENCY TEACUP AND SAUCER", "price_eur": 3.42, "head_noun": "Teacup", "pos_pattern": "PROPN (Head: Teacup)"}, "metrics": {"support": 0.018, "confidence": 0.29, "lift": 3.7}, "linguistic_checks": {"is_packaging_or_stop_item": false, "shares_head_noun": false, "passes_margin_gate": true, "status": "approved_bundle", "rejection_explanation": null}, "promo_mechanics": {"anchor_item": "REGENCY CAKESTAND 3 TIER", "discounted_item": "ROSES REGENCY TEACUP AND SAUCER", "discount_pct": 20, "regular_bundle_price": 18.21, "promo_bundle_price": 17.53, "net_basket_lift_eur": 2.74}, "flyer_copy": {"badge_text": "-20% KOMBI-DEAL", "de": {"title": "Super Spar-Bundle jetzt", "Copy": "Kaufe den Regency Cakestand 3 Tier und erhalte 20% Rabatt auf die Roses Regency Teacup And Saucer!"}, "en": {"title": "Bundle and Save Big", "Copy": "Buy the Regency Cakestand 3 Tier and get 20% off the Roses Regency Teacup And Saucer."}, "el": {"title": "Το τέλειο δίδυμο", "Copy": "Αγοράστε το προϊόν «Regency Cakestand 3 Tier» και κερδίστε 20% έκπτωση στο προϊόν «Roses Regency Teacup And Saucer»."}}}, {"id": "rule-10", "domain": "supermarket_grocery", "antecedent": {"name": "Mozzarella 125g", "price_eur": 0.99, "head_noun": "Mozzarella", "pos_pattern": "NOUN (Head: Mozzarella)"}, "consequent": {"name": "Cherrytomaten 250g", "price_eur": 1.49, "head_noun": "Cherrytomate", "pos_pattern": "NOUN (Compound Head: -tomate)"}, "metrics": {"support": 0.025, "confidence": 0.39, "lift": 3.4}, "linguistic_checks": {"is_packaging_or_stop_item": false, "shares_head_noun": false, "passes_margin_gate": true, "status": "approved_bundle", "rejection_explanation": null}, "promo_mechanics": {"anchor_item": "Cherrytomaten 250g", "discounted_item": "Mozzarella 125g", "discount_pct": 20, "regular_bundle_price": 2.48, "promo_bundle_price": 2.28, "net_basket_lift_eur": 0.79}, "flyer_copy": {"badge_text": "-20% KOMBI-DEAL", "de": {"title": "Spar-Angebot für deine Küche", "Copy": "Kaufe Cherrytomaten 250g und erhalte Mozzarella 125g zu 20% Rabatt!"}, "en": {"title": "Great Deal for Your Kitchen", "Copy": "Buy Cherrytomaten 250g and get Mozzarella 125g 20% off!"}, "el": {"title": "Το τέλειο δίδυμο", "Copy": "Αγοράστε το προϊόν «Cherrytomaten 250g» και κερδίστε 20% έκπτωση στο προϊόν «Mozzarella 125g»."}}}, {"id": "rule-11", "domain": "uci_retail", "antecedent": {"name": "SET OF 3 CAKE TINS PANTRY DESIGN", "price_eur": 5.74, "head_noun": "Tin", "pos_pattern": "PROPN (Head: Tin)"}, "consequent": {"name": "PACK OF 72 RETROSPOT CAKE CASES", "price_eur": 0.64, "head_noun": "Case", "pos_pattern": "PROPN + NOUN (Head: Case)"}, "metrics": {"support": 0.016, "confidence": 0.31, "lift": 3.2}, "linguistic_checks": {"is_packaging_or_stop_item": false, "shares_head_noun": false, "passes_margin_gate": false, "status": "rejected_low_margin", "rejection_explanation": "Low basket expansion: bundle €6.25 is below 110% of the anchor price (€6.31)"}, "promo_mechanics": {"anchor_item": "SET OF 3 CAKE TINS PANTRY DESIGN", "discounted_item": "PACK OF 72 RETROSPOT CAKE CASES", "discount_pct": 20, "regular_bundle_price": 6.38, "promo_bundle_price": 6.25, "net_basket_lift_eur": 0.51}, "flyer_copy": null}, {"id": "rule-12", "domain": "supermarket_grocery", "antecedent": {"name": "Bio Vollmilch 3,8% 1L", "price_eur": 1.29, "head_noun": "Vollmilch", "pos_pattern": "NOUN (Compound Head: -milch)"}, "consequent": {"name": "Bio Fettarme Milch 1,5% 1L", "price_eur": 1.19, "head_noun": "Milch", "pos_pattern": "ADJ + NOUN (Head: Milch)"}, "metrics": {"support": 0.041, "confidence": 0.38, "lift": 3.1}, "linguistic_checks": {"is_packaging_or_stop_item": false, "shares_head_noun": true, "passes_margin_gate": true, "status": "rejected_same_head_noun", "rejection_explanation": "Substitute Variant: Shared syntactic/compound head"}, "promo_mechanics": {"anchor_item": "Bio Vollmilch 3,8% 1L", "discounted_item": "Bio Fettarme Milch 1,5% 1L", "discount_pct": 20, "regular_bundle_price": 2.48, "promo_bundle_price": 2.24, "net_basket_lift_eur": 0.95}, "flyer_copy": null}, {"id": "rule-13", "domain": "supermarket_grocery", "antecedent": {"name": "Spaghetti No. 5 500g", "price_eur": 1.29, "head_noun": "Spaghetti", "pos_pattern": "NOUN (Head: Spaghetti)"}, "consequent": {"name": "Tomatensauce Basilikum 400g", "price_eur": 0.99, "head_noun": "Tomatensauce", "pos_pattern": "NOUN + PROPN (Head: Tomatensauce)"}, "metrics": {"support": 0.029, "confidence": 0.34, "lift": 2.6}, "linguistic_checks": {"is_packaging_or_stop_item": false, "shares_head_noun": false, "passes_margin_gate": true, "status": "approved_bundle", "rejection_explanation": null}, "promo_mechanics": {"anchor_item": "Spaghetti No. 5 500g", "discounted_item": "Tomatensauce Basilikum 400g", "discount_pct": 20, "regular_bundle_price": 2.28, "promo_bundle_price": 2.08, "net_basket_lift_eur": 0.79}, "flyer_copy": {"badge_text": "-20% KOMBI-DEAL", "de": {"title": "Spaghetti und Sauce im Angebot", "Copy": "Kaufe Spaghetti No. 5 500g und erhalte Tomatensauce Basilikum 400g 20% günstiger!"}, "en": {"title": "Spaghetti and Sauce Deal", "Copy": "Buy Spaghetti No. 5 500g and get Tomatensauce Basilikum 400g 20% off!"}, "el": {"title": "Προσφορά της εβδομάδας", "Copy": "Αγοράστε το προϊόν «Spaghetti No. 5 500g» και κερδίστε 20% έκπτωση στο προϊόν «Tomatensauce Basilikum 400g»."}}}, {"id": "rule-14", "domain": "supermarket_grocery", "antecedent": {"name": "Pfand Einwegflasche 0,25 €", "price_eur": 0.25, "head_noun": "Pfand", "pos_pattern": "NOUN (Head: Pfand)"}, "consequent": {"name": "Mineralwasser Classic 1,5L", "price_eur": 0.49, "head_noun": "Mineralwasser", "pos_pattern": "NOUN (Compound Head: -wasser)"}, "metrics": {"support": 0.061, "confidence": 0.72, "lift": 2.4}, "linguistic_checks": {"is_packaging_or_stop_item": true, "shares_head_noun": false, "passes_margin_gate": true, "status": "rejected_packaging_noise", "rejection_explanation": "Packaging/operational noise: \"Pfand\" in \"Pfand Einwegflasche 0,25 €\""}, "promo_mechanics": {"anchor_item": "Mineralwasser Classic 1,5L", "discounted_item": "Pfand Einwegflasche 0,25 €", "discount_pct": 20, "regular_bundle_price": 0.74, "promo_bundle_price": 0.69, "net_basket_lift_eur": 0.2}, "flyer_copy": null}, {"id": "rule-15", "domain": "supermarket_grocery", "antecedent": {"name": "Plastic Carrier Bag", "price_eur": 0.3, "head_noun": "Bag", "pos_pattern": "ADJ + PROPN (Head: Bag)"}, "consequent": {"name": "Bio Bananen 1kg", "price_eur": 1.49, "head_noun": "Banane", "pos_pattern": "NOUN (Head: Banane)"}, "metrics": {"support": 0.088, "confidence": 0.29, "lift": 1.9}, "linguistic_checks": {"is_packaging_or_stop_item": true, "shares_head_noun": false, "passes_margin_gate": false, "status": "rejected_packaging_noise", "rejection_explanation": "Packaging/operational noise: \"Carrier\" in \"Plastic Carrier Bag\""}, "promo_mechanics": {"anchor_item": "Bio Bananen 1kg", "discounted_item": "Plastic Carrier Bag", "discount_pct": 20, "regular_bundle_price": 1.79, "promo_bundle_price": 1.73, "net_basket_lift_eur": 0.24}, "flyer_copy": null}, {"id": "rule-16", "domain": "supermarket_grocery", "antecedent": {"name": "Toastbrot 500g", "price_eur": 1.19, "head_noun": "Toastbrot", "pos_pattern": "NOUN (Compound Head: -brot)"}, "consequent": {"name": "Deutsche Markenbutter 250g", "price_eur": 2.29, "head_noun": "Markenbutter", "pos_pattern": "ADJ + NOUN (Compound Head: -butter)"}, "metrics": {"support": 0.044, "confidence": 0.21, "lift": 1.6}, "linguistic_checks": {"is_packaging_or_stop_item": false, "shares_head_noun": false, "passes_margin_gate": false, "status": "rejected_low_margin", "rejection_explanation": "Low lift: 1.60 is below the bundle floor of 2.00"}, "promo_mechanics": {"anchor_item": "Deutsche Markenbutter 250g", "discounted_item": "Toastbrot 500g", "discount_pct": 20, "regular_bundle_price": 3.48, "promo_bundle_price": 3.24, "net_basket_lift_eur": 0.95}, "flyer_copy": null}]} /*EMBED:END*/;
