#!/bin/bash
# =============================================================================
# Neo4j Restore Script
# =============================================================================
# Usage: ./scripts/restore_neo4j.sh [options] <backup_id>
#
# Options:
#   --backup-dir PATH    Backup directory (default: ./backups)
#   --clear              Clear database before restore
#   --dry-run            Validate backup without applying
#   --labels LABELS      Restore only specific node labels (comma-separated)
#   --relations TYPES    Restore only specific relationship types (comma-separated)
#
# Examples:
#   ./scripts/restore_neo4j.sh 20240115_120000_full
#   ./scripts/restore_neo4j.sh --clear 20240115_120000_full
#   ./scripts/restore_neo4j.sh --dry-run 20240115_120000_full
#   ./scripts/restore_neo4j.sh --labels Entity,Chunk 20240115_120000_full
# =============================================================================

set -e

# Default values
BACKUP_DIR="${BACKUP_DIR:-./backups}"
CLEAR_BEFORE="${CLEAR_BEFORE:-false}"
DRY_RUN="${DRY_RUN:-false}"
NODE_LABELS=""
RELATIONSHIP_TYPES=""
BACKUP_ID=""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --backup-dir)
            BACKUP_DIR="$2"
            shift 2
            ;;
        --clear)
            CLEAR_BEFORE="true"
            shift
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        --labels)
            NODE_LABELS="$2"
            shift 2
            ;;
        --relations)
            RELATIONSHIP_TYPES="$2"
            shift 2
            ;;
        --help|-h)
            head -n 22 "$0" | tail -n +2
            exit 0
            ;;
        -*)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
        *)
            BACKUP_ID="$1"
            shift
            ;;
    esac
done

# Validate backup ID
if [ -z "$BACKUP_ID" ]; then
    echo -e "${RED}Error: Backup ID is required${NC}"
    echo "Usage: $0 [options] <backup_id>"
    echo ""
    echo "Available backups:"
    ls -1 "$BACKUP_DIR" 2>/dev/null || echo "  (none found in $BACKUP_DIR)"
    exit 1
fi

# Check backup exists
if [ ! -d "$BACKUP_DIR/$BACKUP_ID" ]; then
    echo -e "${RED}Error: Backup not found: $BACKUP_DIR/$BACKUP_ID${NC}"
    echo ""
    echo "Available backups:"
    ls -1 "$BACKUP_DIR" 2>/dev/null || echo "  (none found)"
    exit 1
fi

echo "=============================================="
echo "Neo4j Restore Script"
echo "=============================================="
echo "Backup ID: $BACKUP_ID"
echo "Backup Directory: $BACKUP_DIR"
echo "Clear Before Restore: $CLEAR_BEFORE"
echo "Dry Run: $DRY_RUN"
if [ -n "$NODE_LABELS" ]; then
    echo "Node Labels Filter: $NODE_LABELS"
fi
if [ -n "$RELATIONSHIP_TYPES" ]; then
    echo "Relationship Types Filter: $RELATIONSHIP_TYPES"
fi
echo "=============================================="

# Confirmation for destructive operations
if [ "$CLEAR_BEFORE" = "true" ] && [ "$DRY_RUN" = "false" ]; then
    echo -e "${YELLOW}WARNING: This will DELETE ALL existing data before restore!${NC}"
    read -p "Are you sure? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        echo "Restore cancelled."
        exit 0
    fi
fi

# Run Python restore
python3 -c "
import asyncio
import sys
sys.path.insert(0, '.')

from src.graph.backup import RecoveryManager

async def main():
    manager = RecoveryManager(backup_dir='${BACKUP_DIR}')

    # Parse filters
    node_labels = '${NODE_LABELS}'.split(',') if '${NODE_LABELS}' else None
    rel_types = '${RELATIONSHIP_TYPES}'.split(',') if '${RELATIONSHIP_TYPES}' else None

    if node_labels:
        node_labels = [l.strip() for l in node_labels if l.strip()]
    if rel_types:
        rel_types = [t.strip() for t in rel_types if t.strip()]

    result = await manager.restore(
        backup_id='${BACKUP_ID}',
        clear_before_restore=${CLEAR_BEFORE},
        dry_run=${DRY_RUN},
        node_labels=node_labels or None,
        relationship_types=rel_types or None,
    )

    if result.success:
        mode = 'VALIDATED' if ${DRY_RUN} else 'RESTORED'
        print(f'SUCCESS: Backup {mode}')
        print(f'  - Nodes: {result.nodes_restored}')
        print(f'  - Relationships: {result.relationships_restored}')
        print(f'  - Duration: {result.duration_seconds:.2f} seconds')
        if result.warnings:
            print(f'  - Warnings: {len(result.warnings)}')
        return 0
    else:
        print(f'FAILED: {result.errors}')
        return 1

exit(asyncio.run(main()))
"

RESTORE_EXIT_CODE=$?

if [ $RESTORE_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}Restore completed successfully!${NC}"
else
    echo -e "${RED}Restore failed!${NC}"
    exit 1
fi

echo "=============================================="
echo "Restore process completed"
echo "=============================================="
