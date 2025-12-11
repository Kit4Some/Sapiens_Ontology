#!/bin/bash
# =============================================================================
# Neo4j Automated Backup Script
# =============================================================================
# Usage: ./scripts/backup_neo4j.sh [options]
#
# Options:
#   --backup-dir PATH    Backup directory (default: ./backups)
#   --format FORMAT      Backup format: cypher|json|graphml|full (default: cypher)
#   --type TYPE          Backup type: full|incremental|schema_only (default: full)
#   --compress           Enable compression (default: true)
#   --retention DAYS     Retention period in days (default: 30)
#   --verify             Verify backup after creation
#   --dry-run            Show what would be done without executing
#
# Examples:
#   ./scripts/backup_neo4j.sh --format full --verify
#   ./scripts/backup_neo4j.sh --type incremental --retention 7
#   ./scripts/backup_neo4j.sh --backup-dir /mnt/backups --compress
# =============================================================================

set -e

# Default values
BACKUP_DIR="${BACKUP_DIR:-./backups}"
BACKUP_FORMAT="${BACKUP_FORMAT:-cypher}"
BACKUP_TYPE="${BACKUP_TYPE:-full}"
COMPRESS="${COMPRESS:-true}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
VERIFY="${VERIFY:-true}"
DRY_RUN="${DRY_RUN:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --backup-dir)
            BACKUP_DIR="$2"
            shift 2
            ;;
        --format)
            BACKUP_FORMAT="$2"
            shift 2
            ;;
        --type)
            BACKUP_TYPE="$2"
            shift 2
            ;;
        --compress)
            COMPRESS="true"
            shift
            ;;
        --no-compress)
            COMPRESS="false"
            shift
            ;;
        --retention)
            RETENTION_DAYS="$2"
            shift 2
            ;;
        --verify)
            VERIFY="true"
            shift
            ;;
        --no-verify)
            VERIFY="false"
            shift
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        --help|-h)
            head -n 25 "$0" | tail -n +2
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "Neo4j Backup Script"
echo "=============================================="
echo "Backup Directory: $BACKUP_DIR"
echo "Format: $BACKUP_FORMAT"
echo "Type: $BACKUP_TYPE"
echo "Compression: $COMPRESS"
echo "Retention: $RETENTION_DAYS days"
echo "Verify: $VERIFY"
echo "Dry Run: $DRY_RUN"
echo "=============================================="

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Generate backup ID
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_ID="${TIMESTAMP}_${BACKUP_TYPE}"

if [ "$DRY_RUN" = "true" ]; then
    echo -e "${YELLOW}[DRY RUN] Would create backup: $BACKUP_ID${NC}"
    exit 0
fi

# Run Python backup script
echo "Starting backup..."

python3 -c "
import asyncio
import sys
sys.path.insert(0, '.')

from src.graph.backup import BackupManager, BackupConfig, BackupFormat, BackupType

async def main():
    config = BackupConfig(
        backup_dir='${BACKUP_DIR}',
        format=BackupFormat('${BACKUP_FORMAT}'),
        backup_type=BackupType('${BACKUP_TYPE}'),
        compress=${COMPRESS},
        retention_days=${RETENTION_DAYS},
        verify_after_backup=${VERIFY},
    )

    manager = BackupManager(config=config)
    result = await manager.create_backup()

    if result.success:
        print(f'SUCCESS: Backup created at {result.backup_path}')
        print(f'  - Nodes: {result.node_count}')
        print(f'  - Relationships: {result.relationship_count}')
        print(f'  - Size: {result.file_size_bytes / 1024 / 1024:.2f} MB')
        print(f'  - Duration: {result.duration_seconds:.2f} seconds')
        if result.verified:
            print('  - Verification: PASSED')
        return 0
    else:
        print(f'FAILED: {result.errors}')
        return 1

exit(asyncio.run(main()))
"

BACKUP_EXIT_CODE=$?

if [ $BACKUP_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}Backup completed successfully!${NC}"

    # Cleanup old backups
    echo "Cleaning up old backups..."
    python3 -c "
import asyncio
import sys
sys.path.insert(0, '.')

from src.graph.backup import BackupManager, BackupConfig

async def cleanup():
    config = BackupConfig(
        backup_dir='${BACKUP_DIR}',
        retention_days=${RETENTION_DAYS},
    )
    manager = BackupManager(config=config)
    result = await manager.cleanup_old_backups()
    print(f'Cleanup: {result[\"deleted_count\"]} backups deleted, {result[\"kept_count\"]} kept')

asyncio.run(cleanup())
"
else
    echo -e "${RED}Backup failed!${NC}"
    exit 1
fi

echo "=============================================="
echo "Backup process completed"
echo "=============================================="
