import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type FormEvent,
  type PointerEvent,
} from "react";
import type {
  AudioEventArtifact,
  AudioTrackArtifact,
  GenerationModel,
  SoundTrack,
  SoundTrackType,
  TimelineRow,
} from "../types";

interface TimelineProps {
  rows: TimelineRow[];
  soundTracks: SoundTrack[];
  audioTracks: AudioTrackArtifact[];
  audioEvents: AudioEventArtifact[];
  assetVersion: number;
  frame: number;
  frameCount: number;
  onSelectFrame: (frame: number) => void;
  onPlayTrackAudio?: (soundTrackId: string, startFrame: number) => void;
  onPlayEventAudio?: (soundEventId: string, startFrame: number) => void;
  onEditSoundEvent?: (
    soundEventId: string,
    startFrame: number,
    endFrame: number,
  ) => void;
  onCreateSoundTrack?: (input: CreateSoundTrackInput) => void;
  onDeleteSoundTrack?: (soundTrackId: string) => void;
  onEditSoundTrackGenerationModel?: (
    soundTrackId: string,
    generationModel: GenerationModel,
  ) => void;
  onCreateSoundEvent?: (soundTrackId: string, startFrame: number) => void;
  onDeleteSoundEvent?: (soundEventId: string) => void;
  onEditSoundEventDescription?: (
    soundEventId: string,
    description: string,
  ) => void;
  onEditSoundEventDetails?: (soundEventId: string) => void;
}

type LaneKind = "scene" | "tracking" | "visual" | "sound" | "audio";

interface TimelineTooltip {
  content: string;
  x: number;
  y: number;
}

export interface CreateSoundTrackInput {
  trackType: SoundTrackType;
  label: string;
  generationModel: GenerationModel;
  canonicalKey: string | null;
}

interface PackedTimelineRow extends TimelineRow {
  depth: number;
}

interface LaneGroup {
  lane: string;
  soundTrack?: SoundTrack;
  soundTrackId?: string;
  audioTrack?: AudioTrackArtifact;
  rows: PackedTimelineRow[];
  depthCount: number;
}

interface ActiveDrag {
  soundEventId: string;
  mode: "move" | "resize-left" | "resize-right";
  startX: number;
  laneWidth: number;
  startFrame: number;
  endFrame: number;
}

const laneToggles: { kind: LaneKind; label: string }[] = [
  { kind: "scene", label: "Scenes" },
  { kind: "tracking", label: "Tracking" },
  { kind: "visual", label: "Visual" },
  { kind: "sound", label: "Sound" },
  { kind: "audio", label: "Audio" },
];

export default function Timeline({
  rows,
  soundTracks,
  audioTracks,
  audioEvents,
  assetVersion,
  frame,
  frameCount,
  onSelectFrame,
  onPlayTrackAudio,
  onPlayEventAudio,
  onEditSoundEvent,
  onCreateSoundTrack,
  onDeleteSoundTrack,
  onEditSoundTrackGenerationModel,
  onCreateSoundEvent,
  onDeleteSoundEvent,
  onEditSoundEventDescription,
  onEditSoundEventDetails,
}: TimelineProps) {
  const [enabledKinds, setEnabledKinds] = useState<Record<LaneKind, boolean>>({
    scene: true,
    tracking: true,
    visual: true,
    sound: true,
    audio: true,
  });
  const dragRef = useRef<ActiveDrag | null>(null);
  const [showCreateTrack, setShowCreateTrack] = useState(false);
  const [trackType, setTrackType] = useState<SoundTrackType>("sfx");
  const [generationModel, setGenerationModel] = useState<GenerationModel>("t2a");
  const [trackLabel, setTrackLabel] = useState("");
  const [canonicalKey, setCanonicalKey] = useState("");
  const [tooltip, setTooltip] = useState<TimelineTooltip | null>(null);
  const audioByTrack = useMemo(
    () => new Map(audioTracks.map((item) => [item.sound_track_id, item])),
    [audioTracks],
  );
  const audioByEvent = useMemo(
    () => new Map(audioEvents.map((item) => [item.sound_event_id, item])),
    [audioEvents],
  );
  const visibleRows = rows.filter((row) => {
    const kind = rowKind(row);
    if (kind === "sound") {
      return enabledKinds.sound;
    }
    return enabledKinds[kind];
  });
  const lanes = groupRows(
    visibleRows,
    soundTracks,
    audioByTrack,
    enabledKinds.sound,
    enabledKinds.audio,
  );
  const timelineEndFrame = Math.max(frameCount, maxRowEnd(rows), 1);
  const maxPlaybackFrame = Math.max(0, timelineEndFrame - 1);
  const clampedFrame = clamp(frame, 0, maxPlaybackFrame);
  const playheadRatio = clampedFrame / Math.max(maxPlaybackFrame, 1);
  const playheadLeft = `${playheadRatio * 100}%`;

  useEffect(() => {
    function handlePointerMove(event: globalThis.PointerEvent) {
      const drag = dragRef.current;
      if (!drag || !onEditSoundEvent) {
        return;
      }
      const deltaFrames = Math.round(
        ((event.clientX - drag.startX) / Math.max(drag.laneWidth, 1)) *
          timelineEndFrame,
      );
      const length = Math.max(1, drag.endFrame - drag.startFrame);
      let nextStart = drag.startFrame;
      let nextEnd = drag.endFrame;

      if (drag.mode === "resize-left") {
        nextStart = clamp(drag.startFrame + deltaFrames, 0, drag.endFrame - 1);
      } else if (drag.mode === "resize-right") {
        nextEnd = clamp(drag.endFrame + deltaFrames, drag.startFrame + 1, timelineEndFrame);
      } else {
        nextStart = clamp(
          drag.startFrame + deltaFrames,
          0,
          Math.max(0, timelineEndFrame - length),
        );
        nextEnd = nextStart + length;
      }

      onEditSoundEvent(drag.soundEventId, nextStart, nextEnd);
    }

    function handlePointerUp() {
      dragRef.current = null;
    }

    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);
    window.addEventListener("pointercancel", handlePointerUp);
    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
      window.removeEventListener("pointercancel", handlePointerUp);
    };
  }, [onEditSoundEvent, timelineEndFrame]);

  function selectFrame(event: PointerEvent<HTMLElement>) {
    if (maxPlaybackFrame <= 0) {
      return;
    }
    const rect = event.currentTarget.getBoundingClientRect();
    const ratio = clamp((event.clientX - rect.left) / rect.width, 0, 1);
    onSelectFrame(Math.round(ratio * maxPlaybackFrame));
  }

  function startSoundEventEdit(
    event: PointerEvent<HTMLDivElement>,
    row: PackedTimelineRow,
  ) {
    if (!row.sound_event_id || !onEditSoundEvent) {
      return;
    }
    const lane = event.currentTarget.closest(".lane-track");
    if (!(lane instanceof HTMLElement)) {
      return;
    }
    const target = event.target instanceof HTMLElement ? event.target : null;
    const handle = target?.dataset.handle;
    dragRef.current = {
      soundEventId: row.sound_event_id,
      mode:
        handle === "left"
          ? "resize-left"
          : handle === "right"
            ? "resize-right"
            : "move",
      startX: event.clientX,
      laneWidth: lane.getBoundingClientRect().width,
      startFrame: row.start_frame,
      endFrame: row.end_frame,
    };
    event.currentTarget.setPointerCapture?.(event.pointerId);
    event.preventDefault();
    event.stopPropagation();
  }

  function submitCreateTrack(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const label = trackLabel.trim();
    if (!label || !onCreateSoundTrack) {
      return;
    }
    onCreateSoundTrack({
      trackType,
      label,
      generationModel: trackType === "speech" ? "tts" : generationModel,
      canonicalKey: canonicalKey.trim() || null,
    });
    setTrackLabel("");
    setCanonicalKey("");
    setShowCreateTrack(false);
  }

  function createEventInLane(event: PointerEvent<HTMLElement>, soundTrackId: string) {
    if (!onCreateSoundEvent) {
      return;
    }
    const rect = event.currentTarget.getBoundingClientRect();
    const ratio = clamp((event.clientX - rect.left) / rect.width, 0, 1);
    onCreateSoundEvent(soundTrackId, Math.round(ratio * timelineEndFrame));
    event.preventDefault();
    event.stopPropagation();
  }


  function showTooltip(content: string, x: number, y: number) {
    setTooltip({ content, ...tooltipPosition(x, y) });
  }

  function moveTooltip(x: number, y: number) {
    setTooltip((current) =>
      current ? { ...current, ...tooltipPosition(x, y) } : current,
    );
  }

  function hideTooltip() {
    setTooltip(null);
  }

  return (
    <section className="timeline-panel">
      <div className="timeline-header">
        <h2>Timeline</h2>
        <div className="timeline-header-actions">
          <span>
            {visibleRows.length}/{rows.length} items
          </span>
          <div className="timeline-toggles" aria-label="Timeline lane filters">
            {laneToggles.map((toggle) => (
              <button
                className={enabledKinds[toggle.kind] ? "active" : ""}
                key={toggle.kind}
                onClick={() =>
                  setEnabledKinds((current) => ({
                    ...current,
                    [toggle.kind]: !current[toggle.kind],
                  }))
                }
                type="button"
              >
                {toggle.label}
              </button>
            ))}
          </div>
          <button
            className={showCreateTrack ? "timeline-add-track active" : "timeline-add-track"}
            disabled={!onCreateSoundTrack}
            onClick={() => setShowCreateTrack((value) => !value)}
            type="button"
          >
            + Sound Track
          </button>
        </div>
      </div>
      {showCreateTrack ? (
        <form className="timeline-create-track" onSubmit={submitCreateTrack}>
          <select
            value={trackType}
            aria-label="Track type"
            onChange={(event) => {
              const nextType = event.target.value as SoundTrackType;
              setTrackType(nextType);
              setGenerationModel(nextType === "speech" ? "tts" : "t2a");
            }}
          >
            <option value="sfx">sfx</option>
            <option value="ambience">ambience</option>
            <option value="speech">speech</option>
          </select>
          <input
            value={trackLabel}
            onChange={(event) => setTrackLabel(event.target.value)}
            aria-label="Track label"
            placeholder="Track label"
          />
          <select
            aria-label="Generation model"
            disabled={trackType === "speech"}
            value={trackType === "speech" ? "tts" : generationModel}
            onChange={(event) =>
              setGenerationModel(event.target.value as GenerationModel)
            }
          >
            {trackType === "speech" ? (
              <option value="tts">TTS</option>
            ) : (
              <>
                <option value="t2a">T2A</option>
                <option value="v2a">V2A</option>
              </>
            )}
          </select>
          <input
            value={canonicalKey}
            onChange={(event) => setCanonicalKey(event.target.value)}
            aria-label="Canonical key"
            placeholder="canonical_key"
          />
          <button type="submit">Create</button>
        </form>
      ) : null}
      <div className="timeline">
        <div className="timeline-content">
          <div className="timeline-playhead-area">
            <div className="playhead-line" style={{ left: playheadLeft }} />
          </div>
          <div className="timeline-ruler-row">
            <div className="timeline-label-spacer" />
            <div
              className="timeline-ruler"
              onPointerDown={selectFrame}
              onPointerMove={(event) => {
                if (event.buttons === 1) {
                  selectFrame(event);
                }
              }}
            />
          </div>
          {lanes.length === 0 ? (
            <div className="empty-lane">Pipeline lanes will appear here.</div>
          ) : (
            lanes.map(({ lane, soundTrack, soundTrackId, audioTrack, rows: laneRows, depthCount }) => (
              <div className="lane" key={lane}>
                <div className="lane-label" title={lane}>
                  <span
                    tabIndex={0}
                    onMouseEnter={(event) => {
                      showTooltip(
                        trackInfoTooltip(soundTrack, lane, audioTrack),
                        event.clientX,
                        event.clientY,
                      );
                    }}
                    onMouseMove={(event) => moveTooltip(event.clientX, event.clientY)}
                    onMouseLeave={hideTooltip}
                    onFocus={(event) => {
                      const rect = event.currentTarget.getBoundingClientRect();
                      showTooltip(
                        trackInfoTooltip(soundTrack, lane, audioTrack),
                        rect.left,
                        rect.bottom,
                      );
                    }}
                    onBlur={hideTooltip}
                  >
                    {lane}
                  </span>
                  {soundTrackId ? (
                    <div className="lane-actions">
                      {soundTrack ? (
                        <select
                          aria-label={`Generation model for ${soundTrack.label}`}
                          className="lane-model-select"
                          disabled={soundTrack.track_type === "speech"}
                          onChange={(event) =>
                            onEditSoundTrackGenerationModel?.(
                              soundTrack.sound_track_id,
                              event.target.value as GenerationModel,
                            )
                          }
                          title={
                            soundTrack.track_type === "speech"
                              ? "Speech tracks always use TTS"
                              : "Track generation model"
                          }
                          value={soundTrack.generation_model}
                        >
                          {soundTrack.track_type === "speech" ? (
                            <option value="tts">TTS</option>
                          ) : (
                            <>
                              <option value="t2a">T2A</option>
                              <option value="v2a">V2A</option>
                            </>
                          )}
                        </select>
                      ) : null}
                      {audioTrack ? (
                        <button
                          className="lane-play-button"
                          title="Play track audio"
                          onClick={(event) => {
                            event.stopPropagation();
                            onPlayTrackAudio?.(soundTrackId, clampedFrame);
                          }}
                          type="button"
                        >
                          play
                        </button>
                      ) : null}
                      {!(enabledKinds.audio && audioTrack) ? (
                        <>
                          <button
                            title="Add sound event"
                            onClick={() => onCreateSoundEvent?.(soundTrackId, clampedFrame)}
                            type="button"
                          >
                            +
                          </button>
                          <button
                            title="Delete sound track"
                            onClick={() => onDeleteSoundTrack?.(soundTrackId)}
                            type="button"
                          >
                            x
                          </button>
                        </>
                      ) : null}
                    </div>
                  ) : null}
                </div>
                <div
                  className="lane-track"
                  onMouseEnter={(event) => {
                    if (enabledKinds.audio && audioTrack) {
                      showTooltip(
                        audioTrackTooltip(lane, audioTrack),
                        event.clientX,
                        event.clientY,
                      );
                    }
                  }}
                  onMouseMove={(event) => moveTooltip(event.clientX, event.clientY)}
                  onMouseLeave={hideTooltip}
                  onPointerDown={(event) => {
                    if (event.detail === 2 && soundTrackId) {
                      createEventInLane(event, soundTrackId);
                      return;
                    }
                    selectFrame(event);
                  }}
                  onPointerMove={(event) => {
                    if (event.buttons === 1 && !dragRef.current) {
                      selectFrame(event);
                    }
                  }}
                  style={{ minHeight: `${depthCount * 24 + 8}px` }}
                >
                  {enabledKinds.audio && audioTrack ? (
                    <Waveform peaks={audioTrack.waveform_peaks} />
                  ) : null}
                  {laneRows.map((row, index) => {
                    const kind = rowKind(row);
                    const bar = barStyle(row, timelineEndFrame, kind);
                    const editable = isEditableSoundRow(row) && !!onEditSoundEvent;
                    const hasEventAudio = !!row.sound_event_id && audioByEvent.has(row.sound_event_id);
                    const visualStyle =
                      kind === "visual"
                        ? ({ "--bar-color": visualEventColor(row.kind) } as CSSProperties)
                        : {};
                    const genPrefix = row.generation_model
                      ? `[${row.generation_model.toUpperCase()}] `
                      : "";
                    const tooltipContent = soundEventTooltip(row, genPrefix, hasEventAudio);
                    return (
                      <div
                        className={`bar bar-${kind} bar-${safeClass(row.kind)}${
                          editable ? " bar-editable" : ""
                        }${hasEventAudio ? " bar-has-audio" : ""}`}
                        key={
                          row.sound_event_id ??
                          `${lane}-${row.start_frame}-${row.end_frame}-${index}`
                        }
                        onMouseEnter={(event) => {
                          event.stopPropagation();
                          showTooltip(tooltipContent, event.clientX, event.clientY);
                        }}
                        onMouseMove={(event) => {
                          event.stopPropagation();
                          moveTooltip(event.clientX, event.clientY);
                        }}
                        onMouseLeave={(event) => {
                          event.stopPropagation();
                          hideTooltip();
                        }}
                        onFocus={(event) => {
                          const rect = event.currentTarget.getBoundingClientRect();
                          showTooltip(tooltipContent, rect.left, rect.bottom);
                        }}
                        onBlur={hideTooltip}
                        onPointerDown={
                          editable
                            ? (event) => startSoundEventEdit(event, row)
                            : undefined
                        }
                        style={{
                          left: `${bar.left}%`,
                          width: `${bar.width}%`,
                          top: `${row.depth * 24 + 4}px`,
                          ...visualStyle,
                        }}
                        title={`${genPrefix}${row.label}: ${row.start_frame}-${row.end_frame}${hasEventAudio ? " audio ready" : ""}`}
                        onDoubleClick={
                          editable && onEditSoundEventDetails
                            ? (event) => {
                                event.stopPropagation();
                                onEditSoundEventDetails(row.sound_event_id!);
                              }
                            : undefined
                        }
                      >
                        {editable && onDeleteSoundEvent ? (
                          <span className="bar-actions">
                            {hasEventAudio && onPlayEventAudio ? (
                              <button
                                className="bar-action-button"
                                onClick={(event) => {
                                  event.stopPropagation();
                                  onPlayEventAudio(row.sound_event_id ?? "", row.start_frame);
                                }}
                                onPointerDown={(event) => event.stopPropagation()}
                                title="Play event audio"
                                type="button"
                              >
                                p
                              </button>
                            ) : null}
                            {onEditSoundEventDescription ? (
                              <button
                                className="bar-action-button"
                                onClick={(event) => {
                                  event.stopPropagation();
                                  const nextDescription = window.prompt(
                                    "Edit sound event description",
                                    row.label,
                                  );
                                  const trimmed = nextDescription?.trim();
                                  if (trimmed) {
                                    onEditSoundEventDescription(
                                      row.sound_event_id ?? "",
                                      trimmed,
                                    );
                                  }
                                }}
                                onPointerDown={(event) => event.stopPropagation()}
                                title="Edit description"
                                type="button"
                              >
                                e
                              </button>
                            ) : null}
                            <button
                              className="bar-action-button"
                              onClick={(event) => {
                                event.stopPropagation();
                                onDeleteSoundEvent(row.sound_event_id ?? "");
                              }}
                              onPointerDown={(event) => event.stopPropagation()}
                              title="Delete sound event"
                              type="button"
                            >
                              x
                            </button>
                          </span>
                        ) : null}
                        {editable ? (
                          <span
                            aria-hidden="true"
                            className="bar-resize-handle bar-resize-left"
                            data-handle="left"
                          />
                        ) : null}
                        {bar.width < 2 ? "" : bar.width < 7 ? row.kind : row.label}
                        {editable ? (
                          <span
                            aria-hidden="true"
                            className="bar-resize-handle bar-resize-right"
                            data-handle="right"
                          />
                        ) : null}
                      </div>
                    );
                  })}
                </div>
              </div>
            ))
          )}
        </div>
      </div>
      {tooltip ? (
        <div
          className="timeline-tooltip"
          style={{ left: `${tooltip.x}px`, top: `${tooltip.y}px` }}
        >
          {tooltip.content}
        </div>
      ) : null}
    </section>
  );
}

function tooltipPosition(x: number, y: number): { x: number; y: number } {
  const margin = 16;
  const width = 420;
  const height = 180;
  const viewportWidth = window.innerWidth || width + margin * 2;
  const viewportHeight = window.innerHeight || height + margin * 2;
  return {
    x: clamp(x + 14, margin, Math.max(margin, viewportWidth - width - margin)),
    y: clamp(y + 14, margin, Math.max(margin, viewportHeight - height - margin)),
  };
}

function trackInfoTooltip(
  track: SoundTrack | undefined,
  lane: string,
  audioTrack: AudioTrackArtifact | undefined,
): string {
  const lines = [lane];
  if (track) {
    lines.push(`Track type: ${track.track_type}`);
    lines.push(`Generation model: ${track.generation_model}`);
    if (track.canonical_key) {
      lines.push(`Canonical key: ${track.canonical_key}`);
    }
  }
  if (audioTrack) {
    lines.push("Audio: generated track stem");
    lines.push(`Duration: ${audioTrack.duration_sec.toFixed(2)}s`);
    lines.push(`Events: ${audioTrack.event_count}`);
  } else {
    lines.push("Audio: not generated");
  }
  return lines.join("\n");
}

function audioTrackTooltip(lane: string, artifact: AudioTrackArtifact): string {
  const label = artifact.track_label || lane;
  const type = artifact.track_type ? `Type: ${artifact.track_type}` : null;
  const duration = `Duration: ${artifact.duration_sec.toFixed(2)}s`;
  const events = `Events: ${artifact.event_count}`;
  return [label, type, duration, events, "Generated track stem"].filter(Boolean).join("\n");
}

function soundEventTooltip(
  row: TimelineRow,
  generationPrefix: string,
  hasEventAudio: boolean,
): string {
  const generation = generationPrefix
    ? `Generation: ${generationPrefix.replace(/[\[\] ]/g, "")}`
    : "Generation: not assigned";
  return [
    row.label,
    `Frames: ${row.start_frame}-${row.end_frame}`,
    generation,
    hasEventAudio ? "Audio: ready" : "Audio: not generated",
  ].join("\n");
}

function Waveform({ peaks }: { peaks: number[] }) {
  if (peaks.length === 0) {
    return <div className="waveform waveform-empty" />;
  }
  const sampledPeaks = peaks.length > 384 ? samplePeaks(peaks, 384) : peaks;
  return (
    <div className="waveform" aria-hidden="true">
      {sampledPeaks.map((peak, index) => (
        <span
          className="waveform-bar"
          key={index}
          style={{ height: `${Math.max(3, peak * 100)}%` }}
        />
      ))}
    </div>
  );
}

function groupRows(
  rows: TimelineRow[],
  soundTracks: SoundTrack[],
  audioByTrack: Map<string, AudioTrackArtifact>,
  showSound: boolean,
  showAudio: boolean,
): LaneGroup[] {
  const lanes = new Map<string, TimelineRow[]>();
  for (const row of rows) {
    if (rowKind(row) === "sound") {
      continue;
    }
    if (!lanes.has(row.lane)) {
      lanes.set(row.lane, []);
    }
    lanes.get(row.lane)?.push(row);
  }
  for (const track of soundTracks) {
    const trackId = track.sound_track_id;
    if (!showSound && !(showAudio && audioByTrack.has(trackId))) {
      continue;
    }
    const lane = soundLane(track);
    if (!lanes.has(lane)) {
      lanes.set(lane, []);
    }
  }
  for (const row of rows) {
    if (rowKind(row) !== "sound") {
      continue;
    }
    if (!lanes.has(row.lane)) {
      lanes.set(row.lane, []);
    }
    lanes.get(row.lane)?.push(row);
  }
  return [...lanes.entries()].map(([lane, laneRows]) => {
    const packedRows = packLaneRows(laneRows);
    const depthCount =
      packedRows.reduce((maxDepth, row) => Math.max(maxDepth, row.depth), 0) + 1;
    const soundTrack = soundTracks.find((track) => soundLane(track) === lane);
    const soundTrackId = soundTrack?.sound_track_id;
    return {
      lane,
      soundTrack,
      soundTrackId,
      audioTrack: soundTrackId ? audioByTrack.get(soundTrackId) : undefined,
      rows: packedRows,
      depthCount,
    };
  });
}

function soundLane(track: SoundTrack): string {
  return `[${track.track_type}] ${track.label}`;
}

function packLaneRows(rows: TimelineRow[]): PackedTimelineRow[] {
  const depthEndFrames: number[] = [];
  return [...rows]
    .sort(
      (left, right) =>
        left.start_frame - right.start_frame || left.end_frame - right.end_frame,
    )
    .map((row) => {
      const depth = depthEndFrames.findIndex((endFrame) => endFrame <= row.start_frame);
      const nextDepth = depth === -1 ? depthEndFrames.length : depth;
      depthEndFrames[nextDepth] = row.end_frame;
      return { ...row, depth: nextDepth };
    });
}

function rowKind(row: TimelineRow): Exclude<LaneKind, "audio"> {
  if (row.kind === "scene") {
    return "scene";
  }
  if (row.kind === "track") {
    return "tracking";
  }
  if (row.lane.startsWith("visual: ")) {
    return "visual";
  }
  return "sound";
}

function isEditableSoundRow(row: TimelineRow): boolean {
  return rowKind(row) === "sound" && !!row.sound_event_id;
}

function maxRowEnd(rows: TimelineRow[]): number {
  return rows.reduce((maxEnd, row) => Math.max(maxEnd, row.end_frame), 0);
}

function safeClass(value: string): string {
  return value.replace(/[^a-z0-9_-]/gi, "-").toLowerCase();
}

function barStyle(
  row: TimelineRow,
  timelineEndFrame: number,
  kind: Exclude<LaneKind, "audio">,
): { left: number; width: number } {
  const start = clamp(row.start_frame, 0, timelineEndFrame);
  const end = clamp(row.end_frame, start, timelineEndFrame);
  const left = (start / timelineEndFrame) * 100;
  const rawWidth = ((end - start) / timelineEndFrame) * 100;
  const minWidth = kind === "sound" ? 0.12 : 0.4;
  const width = Math.min(100 - left, Math.max(minWidth, rawWidth));
  return { left, width };
}

function samplePeaks(peaks: number[], targetCount: number): number[] {
  if (peaks.length <= targetCount) {
    return peaks;
  }
  const sampled: number[] = [];
  for (let index = 0; index < targetCount; index += 1) {
    const start = Math.floor((index / targetCount) * peaks.length);
    const end = Math.max(start + 1, Math.floor(((index + 1) / targetCount) * peaks.length));
    sampled.push(Math.max(...peaks.slice(start, end)));
  }
  return sampled;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function visualEventColor(kind: string): string {
  const colors: Record<string, string> = {
    contact: "#cf4e4e",
    collision: "#d55f38",
    occlusion: "#8d6fd1",
    motion: "#2f8f72",
    interaction: "#c2872f",
    gesture: "#3f7bd4",
    state_change: "#b64f9f",
  };
  return colors[kind] ?? `hsl(${hashKind(kind)} 68% 42%)`;
}

function hashKind(value: string): number {
  let hash = 0;
  for (const char of value) {
    hash = (hash * 31 + char.charCodeAt(0)) % 360;
  }
  return hash;
}
