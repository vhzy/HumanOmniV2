import sys
import os
import pandas as pd

# Mock Config as before
from types import SimpleNamespace
if "config" not in sys.modules:
    cfg = SimpleNamespace()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cfg.EMOTION_WHEEL_ROOT = os.path.join(current_dir, "emotion_wheel")
    sys.modules["config"] = cfg

sys.path.append("HumanOmniV2/affect_r1")

try:
    from merbench.affectgpt_local.wheel_metrics import _map_label, _raw_mapping, _format_mapping
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from merbench.affectgpt_local.wheel_metrics import _map_label, _raw_mapping, _format_mapping

def test_synonym_mapping():
    print("--- Testing Synonym Mapping Deep Dive ---")
    
    # 1. Check if Format CSV is loaded
    fmt_map = _format_mapping()
    print(f"Format Map loaded: {len(fmt_map)} entries")
    if "happy" in fmt_map:
        print(f"Sample Format 'happy': {fmt_map['happy']}")
    
    # 2. Check if Synonym Excel is loaded
    raw_map = _raw_mapping()
    print(f"Synonym Map loaded: {len(raw_map)} entries")
    
    # Find a synonym pair to test
    # We need to find a word X that maps to Y in raw_map
    test_pair = None
    for k, v in raw_map.items():
        if k != v[0]: # different
            test_pair = (k, v[0])
            break
            
    if test_pair:
        src, tgt = test_pair
        print(f"Found Synonym Pair in map: '{src}' -> '{tgt}'")
        
        # Now test _map_label with this src
        # We use a generic metric like case3_wheel1_level1
        # If src maps to tgt, and tgt is in wheel, it should work.
        
        mapped = _map_label(src, "case3_wheel1_level1")
        print(f"Mapping '{src}' using case3_wheel1_level1 -> '{mapped}'")
        
        if mapped:
            print("SUCCESS: Synonym mapping is active within _map_label")
        else:
            print("WARNING: Synonym mapped but maybe target not in Wheel 1 Level 1?")
            
    else:
        print("Could not find distinct synonym pair in map (maybe empty or identity?)")

if __name__ == "__main__":
    test_synonym_mapping()

