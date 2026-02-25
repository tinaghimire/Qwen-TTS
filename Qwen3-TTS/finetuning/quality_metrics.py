#!/usr/bin/env python3
# coding=utf-8
"""
TTS Evaluation Metrics.

This module provides the following evaluation metrics:
- perplexity (acoustic proxy: how well generated audio fits reference distribution)
- speaker_embedding_consistency
- pronunciation_accuracy
- tonal_accuracy
- prosody_accuracy
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
import librosa


# The five evaluation metrics exposed by this module.
EVALUATION_METRICS = [
    "perplexity",
    "speaker_embedding_consistency",
    "pronunciation_accuracy",
    "tonal_accuracy",
    "prosody_accuracy",
]


class QualityMetricsCalculator:
    """Calculate evaluation metrics for TTS."""

    def __init__(self, sample_rate=24000):
        self.sample_rate = sample_rate

    def calculate_all_metrics(
        self,
        generated_audio: np.ndarray,
        reference_audio: np.ndarray,
        generated_text: str = None,
        reference_text: str = None,
    ) -> Dict[str, float]:
        """
        Calculate all evaluation metrics.

        Args:
            generated_audio: Generated numpy audio array
            reference_audio: Reference/target numpy audio array
            generated_text: Generated text (reserved for future use)
            reference_text: Reference text (reserved for future use)

        Returns:
            Dictionary with EVALUATION_METRICS scores (0-1 scale where 1 = best).
            'perplexity' is stored as a 0-1 score (higher = better).
        """
        metrics = {}

        metrics["perplexity"] = self.perplexity_score(
            generated_audio, reference_audio
        )
        metrics["speaker_embedding_consistency"] = self.speaker_embedding_consistency(
            generated_audio, reference_audio
        )
        metrics["pronunciation_accuracy"] = self.pronunciation_accuracy(
            generated_audio, reference_audio
        )
        metrics["tonal_accuracy"] = self.tonal_accuracy(
            generated_audio, reference_audio
        )
        metrics["prosody_accuracy"] = self.prosody_accuracy(
            generated_audio, reference_audio
        )

        metrics["overall_score"] = np.mean(
            [metrics[k] for k in EVALUATION_METRICS]
        )

        return metrics

    def perplexity_score(
        self,
        generated: np.ndarray,
        reference: np.ndarray,
    ) -> float:
        """
        Acoustic perplexity proxy: how well generated audio fits the reference
        mel distribution. Returned as 0-1 score (higher = better).
        """
        gen_mel = librosa.feature.melspectrogram(
            y=generated,
            sr=self.sample_rate,
            n_mels=80,
        )
        ref_mel = librosa.feature.melspectrogram(
            y=reference,
            sr=self.sample_rate,
            n_mels=80,
        )
        gen_mel_db = librosa.power_to_db(gen_mel + 1e-10, ref=1.0)
        ref_mel_db = librosa.power_to_db(ref_mel + 1e-10, ref=1.0)

        # Fit diagonal Gaussian on reference (per band)
        ref_mean = ref_mel_db.mean(axis=1, keepdims=True)
        ref_std = ref_mel_db.std(axis=1, keepdims=True) + 1e-6

        # NLL of generated frames under reference distribution
        diff = (gen_mel_db - ref_mean) / ref_std
        nll = 0.5 * (np.log(2 * np.pi) + 2 * np.log(ref_std) + diff ** 2)
        mean_nll = np.mean(nll)

        ppl = np.exp(min(mean_nll, 50))  # cap to avoid overflow
        score = 1.0 / (1.0 + np.log1p(ppl))
        return float(min(1.0, max(0.0, score)))

    def speaker_embedding_consistency(
        self,
        generated: np.ndarray,
        reference: np.ndarray,
    ) -> float:
        """
        Consistency of speaker identity between generated and reference.
        Uses MFCC statistics as a proxy for speaker embedding.
        """
        gen_mfcc = librosa.feature.mfcc(
            y=generated, sr=self.sample_rate, n_mfcc=13
        )
        ref_mfcc = librosa.feature.mfcc(
            y=reference, sr=self.sample_rate, n_mfcc=13
        )
        gen_stats = np.concatenate([
            gen_mfcc.mean(axis=1),
            gen_mfcc.std(axis=1),
        ])
        ref_stats = np.concatenate([
            ref_mfcc.mean(axis=1),
            ref_mfcc.std(axis=1),
        ])
        similarity = F.cosine_similarity(
            torch.tensor(gen_stats, dtype=torch.float32).unsqueeze(0),
            torch.tensor(ref_stats, dtype=torch.float32).unsqueeze(0),
        ).item()
        return max(0.0, similarity)

    def pronunciation_accuracy(
        self,
        generated: np.ndarray,
        reference: np.ndarray
    ) -> float:
        """
        Measure articulation accuracy via spectral similarity.

        High similarity indicates correct phoneme production.
        """
        # Extract Mel spectrograms
        gen_mel = librosa.feature.melspectrogram(
            y=generated,
            sr=self.sample_rate,
            n_mels=80,
        )
        ref_mel = librosa.feature.melspectrogram(
            y=reference,
            sr=self.sample_rate,
            n_mels=80,
        )

        # Convert to log scale
        gen_mel_db = librosa.power_to_db(gen_mel, ref=np.max)
        ref_mel_db = librosa.power_to_db(ref_mel, ref=np.max)

        # Pad to match lengths
        min_len = min(gen_mel_db.shape[1], ref_mel_db.shape[1])
        gen_mel_db = gen_mel_db[:, :min_len]
        ref_mel_db = ref_mel_db[:, :min_len]

        # Calculate cosine similarity
        gen_flat = gen_mel_db.flatten()
        ref_flat = ref_mel_db.flatten()

        similarity = F.cosine_similarity(
            torch.tensor(gen_flat).unsqueeze(0),
            torch.tensor(ref_flat).unsqueeze(0)
        ).item()

        return max(0, similarity)  # Normalize to [0, 1]

    def tonal_accuracy(
        self,
        generated: np.ndarray,
        reference: np.ndarray
    ) -> float:
        """
        Measure pitch contour similarity for tonal accuracy.

        Important for Hausa where tone affects meaning.
        """
        # Extract pitch contours
        gen_pitch = self._extract_pitch(generated)
        ref_pitch = self._extract_pitch(reference)

        if len(gen_pitch) == 0 or len(ref_pitch) == 0:
            return 0.5  # Neutral if no pitch detected

        # Align lengths
        min_len = min(len(gen_pitch), len(ref_pitch))
        gen_pitch = gen_pitch[:min_len]
        ref_pitch = ref_pitch[:min_len]

        # Normalize to zero mean
        gen_pitch_norm = gen_pitch - np.mean(gen_pitch[gen_pitch > 0])
        ref_pitch_norm = ref_pitch - np.mean(ref_pitch[ref_pitch > 0])

        # Calculate correlation
        correlation = np.corrcoef(gen_pitch_norm, ref_pitch_norm)[0, 1]

        return max(0, (correlation + 1) / 2)  # Normalize to [0, 1]

    def prosody_accuracy(
        self,
        generated: np.ndarray,
        reference: np.ndarray
    ) -> float:
        """
        Measure appropriate pauses, pacing, and sentence flow.
        """
        # Detect pauses (silences)
        gen_pauses = self._detect_pauses(generated)
        ref_pauses = self._detect_pauses(reference)

        # Energy patterns (pacing)
        gen_energy = self._calculate_energy_contour(generated)
        ref_energy = self._calculate_energy_contour(reference)

        # Align lengths
        min_len = min(len(gen_energy), len(ref_energy))
        gen_energy = gen_energy[:min_len]
        ref_energy = ref_energy[:min_len]

        # Energy pattern correlation
        energy_correlation = np.corrcoef(gen_energy, ref_energy)[0, 1]

        # Pause pattern similarity
        pause_similarity = self._compare_pause_patterns(gen_pauses, ref_pauses)

        # Combine
        prosody_score = (
            max(0, energy_correlation) * 0.6 +
            pause_similarity * 0.4
        )

        return min(1.0, max(0.0, prosody_score))

    # ========== Helper Methods ==========

    def _extract_pitch(self, audio: np.ndarray) -> np.ndarray:
        """Extract fundamental frequency (pitch) contour."""
        pitches, magnitudes = librosa.piptrack(
            y=audio,
            sr=self.sample_rate,
            fmin=50,
            fmax=500,
        )

        # Extract pitch by taking the max magnitude bin at each frame
        pitch_contour = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            pitch_contour.append(pitch)

        return np.array(pitch_contour)

    def _detect_pauses(self, audio: np.ndarray) -> List[Tuple[int, int]]:
        """Detect pause (silence) segments in audio."""
        # Use energy threshold for silence detection
        frame_length = 512
        hop_length = 256

        energy = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]

        # Threshold (dynamic, based on median energy)
        threshold = np.median(energy) * 0.2

        # Find silence frames
        silence_frames = energy < threshold

        # Convert to time intervals
        pauses = []
        in_pause = False
        pause_start = None

        for i, is_silence in enumerate(silence_frames):
            time = i * hop_length / self.sample_rate

            if is_silence and not in_pause:
                in_pause = True
                pause_start = time
            elif not is_silence and in_pause:
                in_pause = False
                pauses.append((pause_start, time))

        return pauses

    def _calculate_energy_contour(self, audio: np.ndarray) -> np.ndarray:
        """Calculate energy contour over time."""
        frame_length = 512
        hop_length = 256

        energy = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]

        return energy

    def _compare_pause_patterns(
        self,
        pauses1: List[Tuple[int, int]],
        pauses2: List[Tuple[int, int]]
    ) -> float:
        """Compare two pause patterns for similarity."""
        if len(pauses1) == 0 and len(pauses2) == 0:
            return 1.0

        if len(pauses1) == 0 or len(pauses2) == 0:
            return 0.0

        # Calculate pause durations
        durations1 = [end - start for start, end in pauses1]
        durations2 = [end - start for start, end in pauses2]

        # Calculate statistics
        mean1, std1 = np.mean(durations1), np.std(durations1)
        mean2, std2 = np.mean(durations2), np.std(durations2)

        # Similarity based on statistics
        mean_similarity = 1.0 - min(1.0, abs(mean1 - mean2) / max(mean1, mean2))
        std_similarity = 1.0 - min(1.0, abs(std1 - std2) / max(std1, std2))

        return (mean_similarity + std_similarity) / 2.0

    def print_metrics_report(self, metrics: Dict[str, float]):
        """Print a formatted metrics report."""
        print("\n" + "="*70)
        print("TTS EVALUATION METRICS REPORT")
        print("="*70)

        category_info = {
            "perplexity": ("Perplexity", "Acoustic fit to reference distribution"),
            "speaker_embedding_consistency": ("Speaker Embedding Consistency", "Speaker identity match"),
            "pronunciation_accuracy": ("Pronunciation Accuracy", "Spectral similarity / articulation"),
            "tonal_accuracy": ("Tonal Accuracy", "Pitch contour similarity"),
            "prosody_accuracy": ("Prosody Accuracy", "Pauses, pacing, and flow"),
            "overall_score": ("OVERALL SCORE", "Average of evaluation metrics"),
        }

        for key in EVALUATION_METRICS + ["overall_score"]:
            if key not in metrics:
                continue
            score = metrics[key]
            if key in category_info:
                name, desc = category_info[key]
                rating_visual = "█" * int(score * 10) + "░" * (10 - int(score * 10))
                print(f"\n{name}: {score:.2f}/1.0 ({rating_visual})")
                print(f"   └─ {desc}")

        print("\n" + "="*70)


def calculate_quality_metrics(
    generated_audio: np.ndarray,
    reference_audio: np.ndarray,
    sample_rate: int = 24000,
    print_report: bool = True
) -> Dict[str, float]:
    """
    Convenience function to calculate all evaluation metrics.

    Args:
        generated_audio: Generated audio numpy array
        reference_audio: Reference/target audio numpy array
        sample_rate: Sample rate in Hz
        print_report: Whether to print formatted report

    Returns:
        Dictionary with EVALUATION_METRICS scores (and overall_score)
    """
    calculator = QualityMetricsCalculator(sample_rate)
    metrics = calculator.calculate_all_metrics(generated_audio, reference_audio)

    if print_report:
        calculator.print_metrics_report(metrics)

    return metrics