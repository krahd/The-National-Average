# WE THE ARTISTS 2026 portrait media delivery

This records the completed technical edition of *The National Average*. It does not record an application submission or the artist's final screening approval.

- Source commit: `fbe6fd4176ad81ced3d6999a2432959e6d5b2aa4` (`portrait versio`).
- Preset: `presets/we-the-artists-portrait-2026.json`, production section; seed `20260810`, foundation `required`, audio enabled.
- Master: `/Users/tom/tom-repos/projects/The-National-Average/outputs/video/we-the-artists-portrait-master/the_national_average_latent_world.mp4`.
- Master SHA-256: `91064e0cbc8f704f362b795fdd3def722af4f4c148bd1f3e0214010800703024`; 336,417,908 bytes.
- Upload copy: `/Users/tom/Downloads/WE-THE-ARTISTS-2026/The-National-Average_WE-THE-ARTISTS_9x16.mp4`.
- Upload SHA-256: `141954b265b9e42777d0feb06481809d151cb7c5a964728d0905f38d7f57b10f`; 86,608,598 bytes. Its hash matches the validated encode in the ignored master directory.
- Extracted thumbnail: `/Users/tom/Downloads/WE-THE-ARTISTS-2026/The-National-Average_WE-THE-ARTISTS_9x16_thumbnail.png` (frame at 3 seconds of the upload encode).

Both MP4s have 1080 × 1920 H.264 High/yuv420p video, 24 fps, 2,304 frames, exactly 96.000 seconds, and AAC-LC stereo audio at 48 kHz for exactly 96.000 seconds. The master video averages 27.84 Mb/s; the upload video averages 7.05 Mb/s. The upload file is strictly below the conservative 100,000,000-byte ceiling.

Render command (run from repository root):

```sh
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python scripts/render_latent_flight.py --preset production \
  --preset-file presets/we-the-artists-portrait-2026.json \
  --out-dir outputs/video/we-the-artists-portrait-master
```

Upload encode command (run from repository root):

```sh
ffmpeg -hide_banner -loglevel error -stats \
  -i outputs/video/we-the-artists-portrait-master/the_national_average_latent_world.mp4 \
  -map 0:v:0 -map 0:a:0 -c:v libx264 -preset medium -b:v 7600k \
  -maxrate 8000k -bufsize 16000k -pix_fmt yuv420p \
  -c:a aac -b:a 160k -ar 48000 -movflags +faststart \
  outputs/video/we-the-artists-portrait-master/The-National-Average_WE-THE-ARTISTS_9x16_upload.mp4
```

Validation: `ffprobe` confirms all properties above. Both MP4s decoded from start to end with FFmpeg and no errors. The portrait diagnostic sampled the title, introduction, all nine chapters, the five-average conclusion and both closing propositions; a contact sheet and full-size conclusion frames from the actual master were visually inspected. The upload thumbnail was extracted from the final encode and inspected. The automated upload scan reported no audio silences; it marked only intentional visually sparse opening passages as black or frozen. The source asset manifest records 251 archive entities, 199 comparable entities, computed weighting records, CLIP and Stable Diffusion VAE assets, and SD-VAE status `ok`. Source analytical values were not modified for this edition.

The five particulate averages are intentionally subdued against the dark field; an artist screening at exhibition scale and with sound remains the final perceptual check. Signed-in portal terms, rights attestations, and application submission remain separate decisions. No application has been submitted.
