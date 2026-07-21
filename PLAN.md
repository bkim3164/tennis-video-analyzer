# Tennis Video Analyzer — Project Plan

Upload a clip of a pro tennis player's forehands/backhands → pose-based neural network classifies each stroke, scores it against expert technique, and returns concrete feedback with an annotated video. Full-stack: React frontend, FastAPI backend, videos stored in AWS S3.

**v1 scope:** analyze clips of pro players (practice-court/baseline-view footage where the player fills the frame works best). **Phase 2 (next month):** extend to analyzing your own gameplay.

**Timeline:** 4 weeks, ~1–2 hrs/day · **Training hardware:** laptop (CPU-friendly by design) · **Repo:** [bkim3164/tennis-video-analyzer](https://github.com/bkim3164/tennis-video-analyzer)

---

## Why this architecture

Training a neural net on raw video frames is infeasible on a laptop and unnecessary. Instead:

1. **Pose estimation** (pretrained MediaPipe) converts video → 33 body keypoints per frame. This is the "vision" step, and it's free — no training needed.
2. **Your neural network** operates on keypoint *sequences* (a stroke ≈ 60 frames × 33 keypoints × (x, y, visibility)). These are tiny inputs, so a BiGRU/temporal-CNN trains in minutes on CPU.
3. **THETIS dataset** provides labeled strokes from **31 beginners and 24 experts** across 12 stroke classes — the expert/beginner distinction is *in the labels*. The NN learns both "what stroke is this" and "does this look like an expert hit it."

```
video ──► MediaPipe pose ──► stroke segmentation ──► normalized keypoint sequences
                                                            │
                                        ┌───────────────────┴──────────────────┐
                                        ▼                                      ▼
                              StrokeClassifier (NN)                 ExpertScorer (NN)
                              forehand / backhand / ...             expert-likeness 0–100
                                        └───────────────────┬──────────────────┘
                                                            ▼
                                   joint-angle comparison vs. expert reference curves
                                                            ▼
                                    feedback report + annotated output video
```

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11 | ML ecosystem standard |
| Pose estimation | MediaPipe Pose | Pretrained, fast on CPU, 33 landmarks |
| Video I/O + annotation | OpenCV | Frame extraction, skeleton/feedback overlay |
| Neural nets | PyTorch | BiGRU / 1D-CNN over keypoint sequences |
| Dataset | [THETIS](https://github.com/THETIS-dataset/dataset) | 1,980 RGB videos, 12 stroke classes, beginner vs. expert labels |
| Numerics | NumPy, pandas | Keypoint processing, angle computation |
| Backend | FastAPI + uvicorn | REST API; imports the ML pipeline directly; async + auto OpenAPI docs |
| Video storage | AWS S3 (boto3) | Private bucket; presigned URLs for upload/playback; free tier |
| Job metadata | DynamoDB (free tier) | Analysis job status + results JSON |
| Frontend | React (Vite) + Tailwind | Upload page, results dashboard |
| Charts | Recharts | Joint-angle plots, score gauges |
| Testing | pytest (backend/ML), Vitest (frontend) | Unit tests for preprocessing, model shapes, API |
| Lint/format | ruff (Python), ESLint + Prettier (JS) | Standard tooling both sides |
| CI | GitHub Actions | Lint + tests on every push, both stacks |
| Experiments | TensorBoard | Loss curves, confusion matrices |
| Env | venv + requirements.txt · npm | Simple, reproducible |

### System architecture

```
React (Vite) ──► FastAPI ──► S3 (raw video, annotated video)
     ▲              │  └────► DynamoDB (job status, results JSON)
     │              ▼
     └──── poll /jobs/{id} ── background task runs ML pipeline
```

Upload flow: frontend requests a presigned URL → uploads video straight to S3 (never through the API) → `POST /analyze` kicks off a FastAPI background task → pipeline downloads from S3, runs pose → NN → feedback, writes annotated video back to S3 and results to DynamoDB → frontend polls, then renders the annotated video (presigned GET) + charts. Presigned-URL uploads are a real production pattern — worth a sentence in the README.

## The two neural networks

**1. StrokeClassifier** — supervised classification.
Input: `(T=60, 99)` keypoint sequence → BiGRU(128) ×2 → FC → softmax over stroke classes (start with 4: forehand, backhand, backhand-2h, serve; expand toward all 12 if accuracy holds). Trained on THETIS. Target: >85% test accuracy.

**2. ExpertScorer** — the interesting one.
Same backbone, binary head trained on THETIS's beginner/expert labels. The sigmoid output *is* your 0–100 technique score: "how confident is the model that an expert hit this?" Calibrated per stroke class. This is a legit, defensible NN evaluation — not a heuristic dressed up.

**Feedback layer** (interpretability): compute joint-angle trajectories (elbow, shoulder, hip rotation, knee bend) per swing phase (preparation → contact → follow-through), compare against the expert distribution in THETIS, and report the biggest deviations in plain English ("elbow too bent at contact: 95° vs. expert avg 152°"). The NN gives the score; the angles explain it.

## Repo structure

```
tennis-video-analyzer/
├── README.md              # demo GIF, architecture diagram, quickstart
├── PLAN.md                # this file
├── requirements.txt
├── .github/workflows/ci.yml
├── configs/               # YAML: model + training hyperparams
├── data/                  # gitignored; download script fills it
├── notebooks/             # EDA, error analysis (cleaned up, committed)
├── src/tennis_analyzer/
│   ├── video/             # loading, frame extraction, output annotation
│   ├── pose/              # MediaPipe wrapper, normalization
│   ├── segmentation/      # swing detection from wrist kinematics
│   ├── data/              # THETIS download, Dataset classes
│   ├── models/            # StrokeClassifier, ExpertScorer
│   ├── training/          # train loop, eval, checkpoints
│   ├── feedback/          # joint angles, phase detection, report generation
│   └── cli.py             # analyze <video> from terminal
├── backend/
│   ├── main.py            # FastAPI app, routes
│   ├── s3.py              # presigned URLs, upload/download helpers
│   ├── jobs.py            # background analysis tasks, DynamoDB status
│   └── tests/
├── frontend/              # Vite + React + Tailwind
│   ├── src/components/    # UploadDropzone, VideoPlayer, ScoreGauge, AngleChart
│   ├── src/pages/         # Upload, Results
│   └── src/api.ts         # typed client for the FastAPI backend
├── infra/                 # S3 bucket + DynamoDB setup script or Terraform
├── tests/
└── models/                # trained weights (small enough to commit)
```

---

## Week 1 — Video → poses (the data pipeline)

- [x] **Day 1** — Repo scaffolding: package layout, venv, requirements, ruff, pytest, CI workflow, .gitignore. First green CI run.
- [x] **Day 2** — Video module: load video, extract frames, resample to fixed FPS. Tests for edge cases (portrait video, odd codecs).
- [x] **Day 3** — MediaPipe integration: keypoints per frame, visibility handling, draw skeleton overlay on video (first cool visual — save a clip).
- [ ] **Day 4** — Keypoint normalization: center on hip, scale by torso length, mirror left-handed players. Critical for generalization — document the math in docstrings.
- [ ] **Day 5** — Stroke segmentation: detect swings via wrist-speed peaks + smoothing; extract fixed-length windows around contact. Test on a pro player clip.
- [ ] **Day 6** — THETIS download script + preprocessing: run the pose pipeline over the dataset, cache keypoint sequences as `.npz`. (Kick off, let it run.)
- [ ] **Day 7** — EDA notebook: expert vs. beginner trajectories, class balance, sequence lengths. Sanity-check the cache. **Milestone: video in → clean labeled sequences out.**

## Week 2 — StrokeClassifier

- [ ] **Day 8** — PyTorch `Dataset`/`DataLoader` for cached sequences; **player-level** train/val/test split (no subject leakage — call this out in the README, it's real ML hygiene).
- [ ] **Day 9** — StrokeClassifier model + config-driven hyperparams. Unit-test I/O shapes.
- [ ] **Day 10** — Training loop: checkpointing, early stopping, TensorBoard logging.
- [ ] **Day 11** — First real training run. Baseline comparison: logistic regression on flattened keypoints (proves the NN earns its keep).
- [ ] **Day 12** — Augmentation: temporal jitter, mirroring, keypoint noise. Retrain, compare.
- [ ] **Day 13** — Error-analysis notebook: confusion matrix, watch misclassified clips, tune.
- [ ] **Day 14** — Evaluation report in `docs/`: metrics table, curves. **Milestone: >85% stroke classification accuracy.**

## Week 3 — ExpertScorer + feedback

- [ ] **Day 15** — ExpertScorer: reuse backbone, binary expert/beginner head. Train.
- [ ] **Day 16** — Score calibration per stroke class; sanity-check on pro clips (should score near the expert end) vs. THETIS beginner clips (should score low).
- [ ] **Day 17** — Joint-angle module: elbow/shoulder/hip/knee angles over time. Well-tested, well-commented — portfolio-grade geometry code.
- [ ] **Day 18** — Swing-phase detection (preparation/contact/follow-through) + expert reference curves per phase from THETIS experts.
- [ ] **Day 19** — Feedback generator: top-3 deviations from expert reference → plain-English tips with numbers.
- [ ] **Day 20** — End-to-end CLI: `tennis-analyzer analyze my_clip.mp4` → annotated video + markdown report.
- [ ] **Day 21** — Test on fresh pro player clips (different players, courts, camera angles); fix segmentation/scoring issues found in the wild. **Milestone: full pipeline works on real pro footage.**

## Week 4 — Backend + frontend, polish, ship

- [ ] **Day 22** — AWS setup (`infra/`): S3 bucket (private, CORS for presigned uploads), DynamoDB table, IAM user with minimal permissions. FastAPI skeleton: `POST /uploads/presign`, health check. Test presigned upload with curl.
- [ ] **Day 23** — Analysis endpoints: `POST /analyze` (background task: S3 download → ML pipeline → annotated video to S3 + results to DynamoDB), `GET /jobs/{id}`. API tests with moto (mocked AWS).
- [ ] **Day 24** — Frontend scaffold: Vite + React + Tailwind, upload dropzone → presigned S3 upload → job polling with progress state.
- [ ] **Day 25** — Results dashboard: annotated video player (presigned GET), per-stroke score gauges, Recharts joint-angle plots vs. expert reference, feedback cards.
- [ ] **Day 26** — End-to-end integration on real pro clips; error states (bad video, no strokes detected); UI polish.
- [ ] **Day 27** — README overhaul: demo GIF, both architecture diagrams, quickstart (incl. AWS setup), results table, limitations section. Docstrings + type hints sweep, final ruff/pytest/ESLint pass.
- [ ] **Day 28** — Record demo GIF; tag `v1.0.0`. Buffer overflow-day if anything slipped.

> Week 4 is dense. If it slips, the CLI (Day 20) already proves the ML end-to-end — the web stack can spill into a few extra days without breaking the project. Optional stretch: deploy frontend to S3/CloudFront or Vercel and backend to a small EC2 instance.

## Git & commit workflow

Recruiters and interviewers *do* look at commit history — a healthy one is part of the deliverable.

- **Commit incrementally, not in bulk.** One commit per logical unit (a module, a test file, a bug fix) — typically 2–4 commits per coding day, never one giant end-of-week dump.
- **Conventional commit messages:** `feat: add wrist-speed stroke segmentation`, `test: cover left-handed mirroring`, `fix: handle portrait video rotation`, `docs: add architecture diagram`. Message body explains *why* when it isn't obvious.
- **Push daily.** Consistent daily activity across 4 weeks reads far better than sporadic bursts.
- **Branch discipline:** work on short-lived feature branches (`feat/segmentation`), merge to `main` when tests pass. Even solo, this shows real workflow habits.
- **Never commit secrets:** AWS keys live in `.env` (gitignored) / `~/.aws/credentials` only. Add a `.env.example`.
- **Tag milestones:** `v0.1` end of week 1 pipeline, `v0.2` trained classifier, `v0.3` CLI, `v1.0.0` full app.

## Risks & mitigations

- **THETIS videos are 640×480 Kinect-era** → MediaPipe still tracks fine; test early (Day 6).
- **Broadcast pro footage is the hard case** → far-court camera angles make the player small in frame, and match broadcasts show two players plus crowd (MediaPipe tracks one person). Prefer practice-court/baseline-view clips where the player fills the frame; add a crop/person-selection step if needed. This is also why "analyze your own gameplay" (phase 2) may end up *easier* — you control the camera.
- **Stroke segmentation is the flakiest part** → keep a manual "trim to one stroke" fallback in the app.
- **Domain gap (THETIS lab setting vs. real courts)** → normalization (Day 4) is the defense; validate on pro clips at every milestone.
- **12-class accuracy disappoints** → collapsing to 4–6 classes is fine and still impressive.
- **AWS bill anxiety** → S3 + DynamoDB free tiers cover this easily at personal scale; set a $5 billing alarm on day one and add a lifecycle rule to auto-delete uploads after 30 days.
- **Week 4 is packed** → the Day 20 CLI is the safety net; the web stack can spill over without breaking the milestone chain.

## Resume bullets this project earns

- Built a full-stack tennis stroke analysis app: React/TypeScript frontend, FastAPI backend with S3 presigned-URL video uploads and DynamoDB job tracking, and PyTorch neural networks (stroke classification at X%, expert-technique scoring) trained on 1,900+ labeled videos.
- Designed a pose-sequence ML pipeline (MediaPipe, OpenCV) with player-level data splits, augmentation, and baseline comparisons; CI-tested (pytest, GitHub Actions) and fully documented.
