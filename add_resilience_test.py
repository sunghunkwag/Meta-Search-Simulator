import sys

with open("UNIFIED_RSI_EXTENDED.py", "r") as f:
    lines = f.readlines()

new_cmd = r'''
def cmd_resilience_test():
    print("[Resilience] Starting Infinite Loop Resilience Test...")
    controller = InventionMetaController()

    # Inject POISON: A strategy that always loops forever
    def poison_strategy(_):
        return "while True:\n    pass"

    # Inject VALID: A strategy that relies on Z3 (standard map)
    def valid_strategy(_):
        return "return [__HOLE__(x) for x in task.input]"

    # Overwrite grammar to purely test selection pressure
    # 50% chance of poison
    controller.repr.grammar['strategy'] = [poison_strategy, valid_strategy]
    controller.repr.weights['strategy'] = {0: 1.0, 1: 1.0}

    print("[Resilience] Injected 50/50 Poison/Valid strategies.")

    # Run for 5 generations
    # We expect timeouts, but eventually success
    start = time.time()
    controller.run(generations=5)
    duration = time.time() - start

    # Check if we found a solution
    solved = len(controller.archive) > 0
    print(f"\n[Resilience] Solved: {solved}")
    print(f"[Resilience] Duration: {duration:.2f}s (should be > 0 due to timeouts but < infinite)")

    if solved:
        print("[PASS] System evolved away from infinite loops.")
    else:
        print("[FAIL] System failed to find solution.")
        sys.exit(1)
'''

# Add function before main()
main_idx = -1
for i, line in enumerate(lines):
    if "def main():" in line:
        main_idx = i
        break

if main_idx != -1:
    lines.insert(main_idx, new_cmd + "\n")

    # Add to parser
    # Find `sub.add_parser('evolve')`
    parser_idx = -1
    for i, line in enumerate(lines):
        if "sub.add_parser('evolve')" in line:
            parser_idx = i
            break

    if parser_idx != -1:
        lines.insert(parser_idx + 1, "    sub.add_parser('resilience-test')\n")

    # Add to dispatch
    # Find `elif args.cmd == 'evolve':`
    dispatch_idx = -1
    for i, line in enumerate(lines):
        if "elif args.cmd == 'evolve':" in line:
            dispatch_idx = i
            break

    if dispatch_idx != -1:
        # We need to insert after the block.
        # Find where the block ends (next elif or else)
        insert_point = dispatch_idx + 3 # approx
        lines.insert(insert_point, "    elif args.cmd == 'resilience-test':\n        cmd_resilience_test()\n")

    with open("UNIFIED_RSI_EXTENDED.py", "w") as f:
        f.writelines(lines)
    print("Added resilience-test command.")
else:
    print("Could not find main")
