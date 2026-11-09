import assert from "node:assert/strict";
import test from "node:test";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";

test("sound lanes keep the sound track order", async (context) => {
  const server = await createServer({
    appType: "custom",
    optimizeDeps: { noDiscovery: true },
    server: { middlewareMode: true },
  });
  context.after(() => server.close());
  const { default: Timeline } = await server.ssrLoadModule(
    "/src/components/Timeline.tsx",
  );
  const soundTracks = [
    {
      sound_track_id: "track-first",
      track_type: "sfx",
      label: "First track",
      generation_model: "t2a",
    },
    {
      sound_track_id: "track-second",
      track_type: "sfx",
      label: "Second track",
      generation_model: "t2a",
    },
  ];
  const rows = [...soundTracks].reverse().map((track, index) => ({
    lane: `[${track.track_type}] ${track.label}`,
    label: `${track.label} event`,
    start_frame: index * 10,
    end_frame: index * 10 + 5,
    kind: track.track_type,
    sound_event_id: `event-${index}`,
    sound_track_id: track.sound_track_id,
    generation_model: track.generation_model,
  }));
  const audioTracks = [...soundTracks].reverse().map((track) => ({
    sound_track_id: track.sound_track_id,
    track_label: track.label,
    track_type: track.track_type,
    path: `${track.sound_track_id}.wav`,
    duration_sec: 1,
    event_count: 1,
    waveform_peaks: [],
  }));

  const html = renderToStaticMarkup(
    React.createElement(Timeline, {
      rows,
      soundTracks,
      audioTracks,
      audioEvents: [],
      assetVersion: 1,
      frame: 0,
      frameCount: 30,
      onSelectFrame() {},
    }),
  );

  assert.ok(
    html.indexOf("[sfx] First track") < html.indexOf("[sfx] Second track"),
  );
});
