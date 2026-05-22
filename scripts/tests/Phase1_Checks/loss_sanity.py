import json, glob, os

# Find latest metrics file
log_dir = "/workspace/outputs/dinov3_vitl_dgx/phase1"
metrics_files = glob.glob(f"{log_dir}/training_metrics*.json") + \
                glob.glob(f"{log_dir}/metrics*.json") + \
                glob.glob(f"{log_dir}/*.json")

# Or read from stdout logs
# Just parse whatever JSON log exists
metrics = []
for f in metrics_files:
    with open(f) as fh:
        for line in fh:
            try: metrics.append(json.loads(line.strip()))
            except: pass

if not metrics:
    print("No metrics file found — check log path")
else:
    metrics.sort(key=lambda x: x['iteration'])
    first, last = metrics[0], metrics[-1]
    
    print(f"Iterations logged : {first['iteration']} → {last['iteration']}")
    print(f"Total loss        : {first['total_loss']:.3f} → {last['total_loss']:.3f}  (↓{first['total_loss']-last['total_loss']:.3f})")
    print(f"iBOT loss         : {first['ibot_loss']:.3f} → {last['ibot_loss']:.3f}")
    print(f"KoLeo             : {first['koleo_loss']:.3f} → {last['koleo_loss']:.3f}  (negative = good)")
    print(f"Backbone grad norm: {first['backbone_grad_norm']:.1f} → {last['backbone_grad_norm']:.3f}  (should be <10)")
    print()
    
    # NaN check
    nans = [m['iteration'] for m in metrics if str(m['total_loss']) == 'nan']
    print(f"NaN losses: {len(nans)} {'✅' if not nans else '❌ at iters: ' + str(nans[:5])}")
    
    # Grad norm stability (last 20 entries)
    recent_grads = [m['backbone_grad_norm'] for m in metrics[-20:]]
    print(f"Recent grad norm range: {min(recent_grads):.3f} – {max(recent_grads):.3f}  {'✅ stable' if max(recent_grads) < 20 else '⚠️ unstable'}")