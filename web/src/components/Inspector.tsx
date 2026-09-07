import { useEffect, useState } from "react";
import { fetchCurrentFrame } from "../api";
import type {
  CurrentFrameRows,
  SceneFrameRow,
  SoundEventFrameRow,
  TimelineRow,
  TrackFrameRow,
  VideoSummary,
  VisualEventFrameRow,
} from "../types";

type InspectorTab = "scene" | "tracks" | "visual" | "sound";

interface InspectorProps {
  video: VideoSummary | null;
  frame: number;
  timelineRows: TimelineRow[];
  version: number;
}

const tabs: { id: InspectorTab; label: string }[] = [
  { id: "scene", label: "Scene" },
  { id: "tracks", label: "Tracks" },
  { id: "visual", label: "Visual" },
  { id: "sound", label: "Sound" },
];

export default function Inspector({
  video,
  frame,
  timelineRows,
  version,
}: InspectorProps) {
  const [rows, setRows] = useState<CurrentFrameRows | null>(null);
  const [activeTab, setActiveTab] = useState<InspectorTab>("scene");
  const sceneRow = activeSceneRow(timelineRows, frame, video?.fps ?? 30);
  const soundRows = activeSoundEventRows(timelineRows, frame, video?.fps ?? 30);

  useEffect(() => {
    if (!video) {
      setRows(null);
      return;
    }
    const controller = new AbortController();
    void fetchCurrentFrame(frame, controller.signal)
      .then(setRows)
      .catch((error: unknown) => {
        if (!(error instanceof DOMException && error.name === "AbortError")) {
          setRows(null);
        }
      });
    return () => controller.abort();
  }, [video, frame, version]);

  return (
    <aside className="inspector">
      <div className="inspector-header">
        <h2>At Frame</h2>
        <span>{frame}</span>
      </div>
      {video ? (
        <>
          <div className="inspector-tabs" role="tablist">
            {tabs.map((tab) => (
              <button
                className={activeTab === tab.id ? "active" : ""}
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                role="tab"
                type="button"
              >
                {tab.label}
              </button>
            ))}
          </div>
          <div className="inspector-body">
            {activeTab === "scene" ? (
              <SceneDetails row={sceneRow ?? rows?.scene ?? null} />
            ) : null}
            {activeTab === "tracks" ? (
              <TrackDetails rows={rows?.tracks ?? []} />
            ) : null}
            {activeTab === "visual" ? (
              <VisualEventDetails rows={rows?.visual_events ?? []} />
            ) : null}
            {activeTab === "sound" ? (
              <SoundEventDetails rows={soundRows.length ? soundRows : rows?.sound_events ?? []} />
            ) : null}
          </div>
        </>
      ) : (
        <p className="muted">No asset loaded.</p>
      )}
    </aside>
  );
}

function activeSoundEventRows(
  rows: TimelineRow[],
  frame: number,
  fps: number,
): SoundEventFrameRow[] {
  return rows
    .filter(
      (row) =>
        row.sound_event_id &&
        row.start_frame <= frame &&
        frame < row.end_frame,
    )
    .map((row) => {
      const match = /^\[(?<trackType>[^ |\]]+)(?: \| [^\]]+)?\] (?<trackLabel>.*)$/.exec(row.lane);
      return {
        sound_event_id: row.sound_event_id,
        sound_track_id: row.sound_track_id ?? "",
        track_label: match?.groups?.trackLabel || row.lane,
        track_type: match?.groups?.trackType || row.kind,
        source: null,
        start_frame: row.start_frame,
        end_frame: row.end_frame,
        duration_sec: Number(((row.end_frame - row.start_frame) / fps).toFixed(2)),
        generation_model: row.generation_model || "t2a",
        description: row.label,
        spoken_text: row.spoken_text ?? null,
        notes: null,
      };
    });
}

function SceneDetails({ row }: { row: SceneFrameRow | null }) {
  if (!row) {
    return <p className="empty-detail">No active scene.</p>;
  }
  return (
    <dl className="detail-grid">
      <Field label="Scene" value={row.scene} />
      <Field label="Frames" value={`${row.start_frame}-${row.end_frame}`} />
      <Field label="Duration" value={`${row.duration_sec}s`} />
    </dl>
  );
}

function TrackDetails({ rows }: { rows: TrackFrameRow[] }) {
  if (rows.length === 0) {
    return <p className="empty-detail">No active tracks.</p>;
  }
  return (
    <div className="detail-list">
      {rows.map((row) => (
        <article className="detail-card" key={`${row.scene}-${row.track}`}>
          <div className="detail-card-title">
            <strong>{row.label}</strong>
            <span>{row.confidence.toFixed(3)}</span>
          </div>
          <dl className="detail-grid">
            <Field label="Scene" value={row.scene} />
            <Field label="Track" value={row.track} />
            <Field label="Bbox" value={formatBbox(row.bbox)} wide />
            <Field label="Mask" value={row.has_mask ? "yes" : "no"} />
          </dl>
        </article>
      ))}
    </div>
  );
}

function VisualEventDetails({ rows }: { rows: VisualEventFrameRow[] }) {
  if (rows.length === 0) {
    return <p className="empty-detail">No visual events at this frame.</p>;
  }
  return (
    <div className="detail-list">
      {rows.map((row, index) => (
        <article
          className="detail-card"
          key={`${row.event_type}-${row.start_frame}-${index}`}
        >
          <div className="detail-card-title">
            <strong>{row.event_type}</strong>
            <span>{row.confidence.toFixed(3)}</span>
          </div>
          <dl className="detail-grid">
            <Field label="Object" value={row.object} />
            <Field label="Related" value={row.related || "-"} />
            <Field label="Frames" value={`${row.start_frame}-${row.end_frame}`} />
            <Field label="Duration" value={`${row.duration_sec}s`} />
            <Field label="Description" value={row.description} wide />
            <Field label="Notes" value={row.notes || "-"} wide />
          </dl>
        </article>
      ))}
    </div>
  );
}

function SoundEventDetails({ rows }: { rows: SoundEventFrameRow[] }) {
  if (rows.length === 0) {
    return <p className="empty-detail">No sound events at this frame.</p>;
  }
  return (
    <div className="detail-list">
      {rows.map((row, index) => (
        <article
          className="detail-card"
          key={`${row.sound_track_id}-${row.start_frame}-${index}`}
        >
          <div className="detail-card-title">
            <strong>{row.track_label}</strong>
            <span>{row.generation_model.toUpperCase()}</span>
          </div>
          <dl className="detail-grid">
            <Field label="Type" value={row.track_type} />
            <Field label="Source" value={row.source || "-"} />
            <Field label="Frames" value={`${row.start_frame}-${row.end_frame}`} />
            <Field label="Duration" value={`${row.duration_sec}s`} />
            <Field label="Description" value={row.description} wide />
            {row.track_type === "speech" ? (
              <Field label="Spoken text" value={row.spoken_text || "-"} wide />
            ) : null}
            <Field label="Notes" value={row.notes || "-"} wide />
          </dl>
        </article>
      ))}
    </div>
  );
}

function Field({
  label,
  value,
  wide = false,
}: {
  label: string;
  value: string | number;
  wide?: boolean;
}) {
  return (
    <div className={wide ? "detail-field wide" : "detail-field"}>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </div>
  );
}

function formatBbox(bbox: number[] | null): string {
  if (!bbox) {
    return "-";
  }
  return bbox.map((value) => value.toFixed(1)).join(", ");
}

function activeSceneRow(
  rows: TimelineRow[],
  frame: number,
  fps: number,
): SceneFrameRow | null {
  const scene = rows.find(
    (row) =>
      row.kind === "scene" &&
      row.start_frame <= frame &&
      frame < row.end_frame,
  );
  if (!scene) {
    return null;
  }
  const match = /^scene (?<scene>\d+)$/.exec(scene.label);
  return {
    scene: match?.groups?.scene ? Number(match.groups.scene) : 0,
    start_frame: scene.start_frame,
    end_frame: scene.end_frame,
    duration_sec: Number(((scene.end_frame - scene.start_frame) / fps).toFixed(2)),
  };
}
