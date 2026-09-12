import { FormEvent, useEffect, useState } from "react";
import {
  fetchAssetSummary,
  cleanAsset,
  importAsset,
  resetSoundTimeline,
  startRun,
  generateAudio,
  deleteEventAudio,
  regenerateEventAudio,
} from "./api";
import type { VideoAsset } from "./types";
import VideoEditor from "./components/VideoEditor";
import type { AssetResponse } from "./types";

const emptyAsset: AssetResponse = {
  status: "idle",
  stage: null,
  error: null,
  version: 0,
  asset_version: 0,
  updated_at: "",
  video: null,
  timeline_rows: [],
  audio_tracks: [],
  audio_events: [],
};

export default function App() {
  const [state, setState] = useState<AssetResponse>(emptyAsset);
  const [submitError, setSubmitError] = useState<string | null>(null);

  useEffect(() => {
    void refreshAssetSummary();
    const events = new EventSource("/events");
    events.addEventListener("asset_update", () => {
      void refreshAssetSummary();
    });
    return () => events.close();
  }, []);

  async function refreshAssetSummary() {
    const nextState = await fetchAssetSummary();
    setState(nextState);
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitError(null);
    try {
      await startRun(event.currentTarget);
      await refreshAssetSummary();
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : String(error));
    }
  }

  async function handleImport(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitError(null);
    try {
      await importAsset(event.currentTarget);
      await refreshAssetSummary();
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : String(error));
    }
  }

  async function handleResetSoundTimeline() {
    setSubmitError(null);
    try {
      await resetSoundTimeline();
      await refreshAssetSummary();
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : String(error));
    }
  }

  async function handleClean() {
    setSubmitError(null);
    try {
      await cleanAsset();
      await refreshAssetSummary();
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : String(error));
      throw error;
    }
  }

  async function handleGenerateAudio(event: FormEvent<HTMLFormElement>, draftAsset: VideoAsset | null) {
    event.preventDefault();
    setSubmitError(null);
    try {
      await generateAudio(event.currentTarget, draftAsset);
      await refreshAssetSummary();
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : String(error));
      throw error;
    }
  }

  async function handleDeleteEventAudio(soundEventId: string) {
    setSubmitError(null);
    try {
      await deleteEventAudio(soundEventId);
      await refreshAssetSummary();
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : String(error));
      throw error;
    }
  }

  async function handleRegenerateEventAudio(
    soundEventId: string,
    description: string,
    spokenText: string | null,
    serverUrl: string | null,
  ) {
    setSubmitError(null);
    try {
      await regenerateEventAudio(soundEventId, description, spokenText, serverUrl);
      await refreshAssetSummary();
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : String(error));
      throw error;
    }
  }

  return (
    <VideoEditor
      state={state}
      submitError={submitError}
      onSubmit={handleSubmit}
      onImport={handleImport}
      onClean={handleClean}
      onResetSoundTimeline={handleResetSoundTimeline}
      onGenerateAudio={handleGenerateAudio}
      onDeleteEventAudio={handleDeleteEventAudio}
      onRegenerateEventAudio={handleRegenerateEventAudio}
    />
  );
}
