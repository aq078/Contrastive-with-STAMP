"""Run from the repository root; legacy experiments remain untouched."""
import argparse
import json
import signal


def stop_on_disconnect(signum, frame):
    # Unwind pool/lock contexts so workers do not keep running as SSH orphans.
    raise SystemExit(128 + signum)


def main():
    signal.signal(signal.SIGHUP, stop_on_disconnect)
    signal.signal(signal.SIGTERM, stop_on_disconnect)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'select', 'fit', 'test', 'all', 'status'])
    parser.add_argument('--config', default='configs/sere_temporal_lopo.json')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--devices', nargs='+',
                        help='Selection/fit worker devices, e.g. cuda:0 cuda:1 cuda:2. '
                             'Prepare/test use the first device. Default: --device.')
    parser.add_argument('--reduction', choices=['none', 'mean'],
                        help='none: P x 99; mean: 3 x 33. Overrides the config reduction.')
    args = parser.parse_args()
    devices = args.devices or [args.device]
    device = devices[0]
    from stamp.sere.workflow import initialize, prepare_portions, select, fit, test
    from stamp.sere.checkpoint import run_lock
    from sere_selection_policy import apply_selection_policy
    from sere_single_seed import select_single_seed, require_single_seed_selection
    from sere_window_evaluation import fit_all_windows, test_all_windows
    config, manifest, output, protocol_hash = initialize(args.config, reduction=args.reduction)
    print(f"{config['cv']['n_folds']}-fold CV; reduction={config['reduction']}; "
          f"devices={devices}; results: {output}", flush=True)
    with run_lock(output / 'orchestrator'):
        config = apply_selection_policy(config, args.config, output, protocol_hash)
        print(f"Candidate window lengths: {config['window_lengths']}", flush=True)
        if config.get('selection_seeds'):
            print(f"Active seeds (CV/final/test): {config['seeds']}; validation BA SD across folds (ddof=1)", flush=True)
        if args.phase == 'status':
            if not config.get('selection_seeds'):
                parser.error('status currently requires a single-seed selection policy')
            result = select_single_seed(config, manifest, output, protocol_hash, devices, report_only=True)
            brief = {length: dict(status=item['status'], methods={
                method: {k: r[k] for k in ('epoch', 'score', 'fold_std_ba', 'fold_scores')}
                for method, r in item['methods'].items()}) for length, item in result['windows'].items()}
            print(json.dumps(brief, indent=2))
        if args.phase in ('prepare', 'all'):
            prepare_portions(config, manifest, output, protocol_hash,
                             config['window_lengths'], 'development', device)
        if args.phase in ('select', 'all'):
            selector = select_single_seed if config.get('selection_seeds') else select
            result = selector(config, manifest, output, protocol_hash, devices)
            print(json.dumps(result['methods'], indent=2))
        if args.phase in ('fit', 'all'):
            if config.get('selection_seeds'):
                require_single_seed_selection(config, output, protocol_hash)
                fit_all_windows(config, manifest, output, protocol_hash, devices)
            else:
                fit(config, manifest, output, protocol_hash, devices)
        if args.phase in ('test', 'all'):
            if config.get('selection_seeds'):
                require_single_seed_selection(config, output, protocol_hash)
                result = test_all_windows(config, manifest, output, protocol_hash, device)
            else:
                result = test(config, manifest, output, protocol_hash, device)
            print(json.dumps(result.get('rows', result.get('methods')), indent=2))


if __name__ == '__main__':
    main()
