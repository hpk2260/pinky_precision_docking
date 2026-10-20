#!/usr/bin/env python3
"""
Sound Effects Module for Pinky Pro Precision Docking & Mission Runner.

Features:
1. Docking Complete Chime: '도-미-솔-도' (C5-E5-G5-C6) ascending chime.
2. Undocking Backup Melody: 'Für Elise' (엘리제를 위하여) - standard AGV/forklift reverse chime.
3. Audio backends supported:
   - ALSA (aplay)
   - PulseAudio (paplay)
   - PipeWire (pw-play)
   - Raspberry Pi GPIO Piezo Buzzer (optional, via gpiozero or RPi.GPIO if hardware present)
   - Terminal Bell (\\a) fallback if sound card is unavailable
"""

import math
import os
import shutil
import struct
import subprocess
import sys
import threading
import time
import wave
from typing import List, Optional, Tuple


CACHE_DIR = os.path.expanduser("~/.cache/pinky_sounds")


def _generate_wav_file(notes: List[Tuple[float, float]], output_path: str,
                       sample_rate: int = 22050, volume: float = 0.5) -> str:
    """Generate a clean synthetic WAV file for given (frequency_hz, duration_sec) pairs."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frames = bytearray()

    for freq, dur in notes:
        num_samples = int(sample_rate * dur)
        if freq <= 0:
            # Silence / rest
            frames.extend(b'\x00\x00' * num_samples)
            continue

        # Soft attack & decay envelope (15ms ramp) to avoid speaker popping
        ramp = min(int(sample_rate * 0.015), max(1, num_samples // 4))
        for i in range(num_samples):
            env = 1.0
            if i < ramp:
                env = i / ramp
            elif i > num_samples - ramp:
                env = (num_samples - i) / ramp

            # Fundamental + slight 2nd harmonic (warmth/bell effect)
            val = (0.85 * math.sin(2.0 * math.pi * freq * (i / sample_rate)) +
                   0.15 * math.sin(4.0 * math.pi * freq * (i / sample_rate))) * volume * env
            sample = int(max(-1.0, min(1.0, val)) * 32767)
            frames.extend(struct.pack('<h', sample))

    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(frames)

    return output_path


# 1. '도-미-솔-도' (C5-E5-G5-C6) ascending chime
DOCKING_CHIME_NOTES: List[Tuple[float, float]] = [
    (523.25, 0.12),  # C5 (도)
    (659.25, 0.12),  # E5 (미)
    (783.99, 0.12),  # G5 (솔)
    (1046.50, 0.35), # C6 (높은 도, 페이드아웃)
]

# 2. 후진 경고음: 표준 산업용 AGV/지게차 단일 비프음 ('삐~ 삐~', 1000Hz)
REVERSE_BEEP_NOTES: List[Tuple[float, float]] = [
    (1000.0, 0.25),  # 1000Hz 비프 250ms
    (0.0,    0.20),  # 무음 200ms 휴지기
]
FUR_ELISE_NOTES = REVERSE_BEEP_NOTES  # 하위 호환성 유지


class SoundManager:
    """Manages audio effects for docking completion and undocking reverse."""

    def __init__(self, piezo_pin: Optional[int] = None):
        self.piezo_pin = piezo_pin or (
            int(os.environ["PINKY_PIEZO_PIN"]) if "PINKY_PIEZO_PIN" in os.environ else None
        )
        self.docking_wav = os.path.join(CACHE_DIR, "docking_complete_chime.wav")
        self.reverse_beep_wav = os.path.join(CACHE_DIR, "reverse_beep.wav")
        self.fur_elise_wav = self.reverse_beep_wav
        self._reverse_proc: Optional[subprocess.Popen] = None
        self._reverse_thread: Optional[threading.Thread] = None
        self._reverse_stop_event = threading.Event()
        self._gpio_buzzer = None

        # Pre-generate WAV files
        self._prepare_audio_files()

        # Find best available player
        self.player_cmd = self._find_player()

        # Init optional GPIO Piezo Buzzer if hardware library present
        if self.piezo_pin is not None:
            self._init_gpio_piezo()

    def _prepare_audio_files(self):
        """Pre-generate WAV audio assets."""
        try:
            if not os.path.exists(self.docking_wav):
                _generate_wav_file(DOCKING_CHIME_NOTES, self.docking_wav, volume=0.55)
            # Regenerate beep wav to apply clean single-tone beep
            _generate_wav_file(REVERSE_BEEP_NOTES, self.reverse_beep_wav, volume=0.50)
        except Exception as e:
            print(f"[SoundManager Warning] Could not pre-generate audio files: {e}")

    def _find_player(self) -> Optional[List[str]]:
        """Find an available sound playback utility."""
        search_paths = [
            "/home/user/.local/bin/aplay",
            "/home/user/.local/bin/paplay",
            "/home/user/.local/bin/pw-play",
            shutil.which("aplay"),
            shutil.which("paplay"),
            shutil.which("pw-play"),
            shutil.which("ffplay"),
        ]
        for p in search_paths:
            if p and os.path.isfile(p) and os.access(p, os.X_OK):
                cmd_name = os.path.basename(p)
                if "aplay" in cmd_name:
                    return [p, "-q"]
                elif "paplay" in cmd_name:
                    return [p]
                elif "pw-play" in cmd_name:
                    return [p]
                elif "ffplay" in cmd_name:
                    return [p, "-nodisp", "-autoexit", "-loglevel", "quiet"]
        return None

    def _init_gpio_piezo(self):
        """Initialize GPIO piezo buzzer on Raspberry Pi if library is available."""
        try:
            from gpiozero import TonalBuzzer  # type: ignore
            self._gpio_buzzer = TonalBuzzer(self.piezo_pin)
            print(f"[SoundManager] Initialized GPIO Piezo TonalBuzzer on pin {self.piezo_pin}")
        except Exception:
            try:
                import RPi.GPIO as GPIO  # type: ignore
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(self.piezo_pin, GPIO.OUT)
                self._gpio_buzzer = ("rpi_gpio", self.piezo_pin)
                print(f"[SoundManager] Initialized RPi.GPIO Piezo Buzzer on pin {self.piezo_pin}")
            except Exception as e:
                print(f"[SoundManager Info] GPIO library not available for piezo: {e}")
                self._gpio_buzzer = None

    def _play_file_subprocess(self, wav_path: str) -> Optional[subprocess.Popen]:
        """Spawn player subprocess or fall back silently."""
        if not self.player_cmd or not os.path.exists(wav_path):
            return None
        try:
            return subprocess.Popen(
                self.player_cmd + [wav_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception:
            return None

    def play_docking_complete(self, blocking: bool = False):
        """
        Play '도-미-솔-도' (C5-E5-G5-C6) ascending chime upon docking completion.
        """
        print("\n🔔 [Sound] 🎶 도-미-솔-도! Precision Docking Complete Chime 🎶")
        # Visual & terminal alert
        sys.stdout.write('\a')
        sys.stdout.flush()

        # Play via speaker
        proc = self._play_file_subprocess(self.docking_wav)

        # Also play via GPIO piezo if configured
        if self._gpio_buzzer:
            threading.Thread(target=self._play_gpio_sequence, args=(DOCKING_CHIME_NOTES,), daemon=True).start()

        if blocking and proc:
            try:
                proc.wait(timeout=2.0)
            except Exception:
                pass

    def start_reverse_melody(self):
        """
        Start playing single-tone beep-beep ('삐~ 삐~') in loop while robot is reversing during undock backup.
        """
        print("🔊 [Sound] 삐~ 삐~ Undocking Reverse Beep Active...")
        self.stop_reverse_melody()
        self._reverse_stop_event.clear()

        def _loop_playback():
            while not self._reverse_stop_event.is_set():
                proc = self._play_file_subprocess(self.reverse_beep_wav)
                self._reverse_proc = proc
                if proc:
                    while proc.poll() is None:
                        if self._reverse_stop_event.is_set():
                            try:
                                proc.terminate()
                                proc.wait(timeout=0.2)
                            except Exception:
                                pass
                            return
                        time.sleep(0.05)
                else:
                    # Fallback rhythm ticks if no soundcard
                    time.sleep(0.5)

        self._reverse_thread = threading.Thread(target=_loop_playback, daemon=True)
        self._reverse_thread.start()

    def stop_reverse_melody(self):
        """Stop reverse melody immediately when backup completes."""
        self._reverse_stop_event.set()
        if self._reverse_proc:
            try:
                self._reverse_proc.terminate()
                self._reverse_proc.wait(timeout=0.3)
            except Exception:
                pass
            self._reverse_proc = None
        print("⏹️  [Sound] Undocking Reverse Melody Stopped.")

    def _play_gpio_sequence(self, notes: List[Tuple[float, float]]):
        """Play melody directly through Raspberry Pi GPIO piezo buzzer."""
        if not self._gpio_buzzer:
            return
        try:
            # gpiozero TonalBuzzer
            if hasattr(self._gpio_buzzer, 'play'):
                for freq, dur in notes:
                    if self._reverse_stop_event.is_set():
                        break
                    if freq > 0:
                        self._gpio_buzzer.play(freq)
                    else:
                        self._gpio_buzzer.stop()
                    time.sleep(dur)
                self._gpio_buzzer.stop()
        except Exception as e:
            print(f"[SoundManager Piezo Error] {e}")


# Global singleton instance
_GLOBAL_SOUND_MANAGER: Optional[SoundManager] = None


def get_sound_manager() -> SoundManager:
    global _GLOBAL_SOUND_MANAGER
    if _GLOBAL_SOUND_MANAGER is None:
        _GLOBAL_SOUND_MANAGER = SoundManager()
    return _GLOBAL_SOUND_MANAGER


def play_docking_complete(blocking: bool = False):
    """Convenience function: Play '도-미-솔-도' chime."""
    get_sound_manager().play_docking_complete(blocking=blocking)


def start_reverse_melody():
    """Convenience function: Start 'Für Elise' reverse chime."""
    get_sound_manager().start_reverse_melody()


def stop_reverse_melody():
    """Convenience function: Stop 'Für Elise' reverse chime."""
    get_sound_manager().stop_reverse_melody()


def main():
    """CLI test tool for Pinky sound effects."""
    print("=" * 60)
    print("        🎵 Pinky Pro Audio Effects Test Tool 🎵")
    print("=" * 60)
    sm = get_sound_manager()
    print(f"Available audio player: {sm.player_cmd}")
    print(f"Docking WAV path:       {sm.docking_wav}")
    print(f"Für Elise WAV path:     {sm.fur_elise_wav}")
    print("\n[1/2] Testing Docking Complete Chime ('도-미-솔-도')...")
    sm.play_docking_complete(blocking=True)
    time.sleep(1.0)

    print("\n[2/2] Testing Undocking Reverse Melody ('Für Elise') for 5 seconds...")
    sm.start_reverse_melody()
    time.sleep(5.0)
    sm.stop_reverse_melody()

    print("\n🎉 Sound effect test finished successfully!")


if __name__ == "__main__":
    main()
