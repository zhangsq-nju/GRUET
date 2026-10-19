import argparse
import json

from agent_uq.uq.reasoning_contender_inputs import prepare_inventory
from agent_uq.uq.reasoning_contender_runtime import preflight, produce_phase


def main(argv=None):
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='action', required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--inventory', required=True)
    prepare.add_argument('--artifact-root', required=True)
    prepare.add_argument('--run-id')
    for action in ('preflight', 'produce'):
        command = commands.add_parser(action)
        command.add_argument('--artifact-root', required=True)
        command.add_argument('--runtime-json', required=True)
        command.add_argument('--prepared-only', action='store_true')
        if action == 'produce':
            command.add_argument('--phase', choices=('topo', 'cot'), required=True)
            command.add_argument('--worker-index', type=int, default=0)
            command.add_argument('--worker-count', type=int, default=1)
            command.add_argument('--turn-filter', help='Path to {run_id: {record_id: [original_turn_idx]}} JSON')
    args = parser.parse_args(argv)
    if args.action == 'prepare':
        report = prepare_inventory(args.inventory, args.artifact_root, run_id=args.run_id)
    else:
        with open(args.runtime_json) as handle:
            runtime = json.load(handle)
        if args.action == 'preflight':
            report = preflight(args.artifact_root, args.artifact_root, runtime=runtime,
                               prepared_only=args.prepared_only)
        else:
            turn_filter = None
            if args.turn_filter:
                with open(args.turn_filter) as handle:
                    turn_filter = json.load(handle)
            report = produce_phase(args.artifact_root, args.artifact_root, runtime=runtime,
                                   phase=args.phase, turn_filter=turn_filter, prepared_only=args.prepared_only,
                                   worker_index=args.worker_index, worker_count=args.worker_count)
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
