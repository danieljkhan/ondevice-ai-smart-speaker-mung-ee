# Jetson Alignment Plan

## Goal

Use a temporary split setup now, then safely converge to a Git-managed `/opt/mungi` later.

Current assumption:

- live directory: `/opt/mungi`
- Git-managed clone: `/opt/mungi-repo`

## Phase 1: Keep the split

Do this now:

1. Keep the current live directory untouched.
2. Clone the GitHub repository to `/opt/mungi-repo`.
3. Use `/opt/mungi-repo` as the development mirror on Jetson.
4. Do not point `systemd` or runtime jobs to `/opt/mungi-repo` yet.

## Phase 2: Create a migration inventory

Run the inventory script from the repository:

```bash
cd /opt/mungi-repo
bash scripts/inventory_runtime.sh /opt/mungi /opt/mungi-repo
```

This produces a dated report under `./reports/`.

Review the differences and classify them into four groups:

- should become source code in Git
- should become runtime data in `/var/lib/mungi`
- should become logs in `/var/log/mungi`
- should remain local-only model artifacts

## Phase 3: Normalize runtime paths

Before switching the live directory, prepare these runtime targets:

```bash
sudo mkdir -p /var/lib/mungi/config
sudo mkdir -p /var/log/mungi
sudo mkdir -p /var/lib/mungi/conversations
```

Move mutable files out of the live source tree once they are identified.

Typical examples:

- `config.json` -> `/var/lib/mungi/config/`
- conversation history -> `/var/lib/mungi/conversations/`
- logs -> `/var/log/mungi/`

## Phase 4: Backup before cutover

Take a full backup before any replacement:

```bash
cd /opt
sudo cp -a mungi mungi-backup-$(date +%Y%m%d)
```

Do not skip this step.

## Phase 5: Promote the Git repository

After classification and backup are complete:

1. stop the service
2. keep the backup
3. replace `/opt/mungi` with the Git-managed tree
4. restore only runtime data into `/var/lib/mungi`
5. restore only logs into `/var/log/mungi`
6. reinstall or verify dependencies
7. restart and validate the service

Suggested cutover outline:

```bash
sudo systemctl stop mungi || true
cd /opt
sudo mv mungi mungi-precutover-$(date +%Y%m%d)
sudo mv mungi-repo mungi
```

Then validate:

- Python environment loads correctly
- CUDA libraries resolve outside interactive shells
- ONNX provider detection works as expected
- service starts under `systemd`
- conversations and config are read from `/var/lib/mungi`

## Phase 6: Clean up

Only after stable validation:

- archive or remove old backup directories
- update runbooks
- point all future development and deployment to the Git-managed `/opt/mungi`

## Immediate next command

If `/opt/mungi-repo` does not exist yet:

```bash
cd /opt
sudo git clone https://github.com/OWNER/ondevice-ai-smart-speaker-mung-ee.git mungi-repo
```

If it already exists:

```bash
cd /opt/mungi-repo
git pull
```
