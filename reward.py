"""
Reward function for Confident Task LoRA training.

This module contains the reward function that evaluates model outputs based on:
1. Confidence tags (<c_math> or <u_math>)
2. Answer correctness (compared to ground truth)

Reward scheme (based on reward.txt):
- C1 (Ideal): Correct + <c_math> = +2.0 (Excellent - encourages correct and confident)
- C2 (Cautious): Correct + <u_math> = +1.0 (Good - correct but not confident)
- C3 (Honest): Wrong + <u_math> = +0.1 (Acceptable - knows it's wrong, avoids hallucination)
- C4 (Hallucination): Wrong + <c_math> = -1.0 (Worst - wrong but confident, strong penalty)
- C5 (Format error): No tag/box = -0.5 (Penalty for not following instructions)
"""

from grader import math_equal
from metrics import extract_box


def reward_fn(generated_text: str, ground_truth: str) -> float:
    """
    Compute reward for a generated response based on confidence and correctness.
    
    Args:
        generated_text: The model's generated response
        ground_truth: The ground truth solution/answer
    
    Returns:
        float: Reward value according to the scheme:
            +2.0: C1 (Ideal) - Correct + <c_math>
            +1.0: C2 (Cautious) - Correct + <u_math>
            +0.1: C3 (Honest) - Wrong + <u_math>
            -1.0: C4 (Hallucination) - Wrong + <c_math>
            -0.5: C5 (Format error) - No tag/box
    """
    # 1. Check for confidence tags
    has_c_math = "<c_math>" in generated_text
    has_u_math = "<u_math>" in generated_text
    
    # 2. Extract predicted answer
    pred_answer = extract_box(generated_text)
    
    # 3. Extract ground truth answer
    gt_answer = extract_box(ground_truth)
    if not gt_answer:
        gt_answer = ground_truth  # Fallback if no box in GT
    
    # 4. Determine correctness
    is_correct = False
    if pred_answer and gt_answer:
        is_correct = math_equal(pred_answer, gt_answer)
    
    # 5. Apply reward scheme based on (correctness, confidence_tag)
    
    # C5: Format error - no confidence tag or no answer box
    if not has_c_math and not has_u_math:
        return -0.5  # No confidence tag
    
    if not pred_answer:
        return -0.5  # No answer in box (format error)
    
    # Now we have a valid answer and confidence tag
    
    # C1 (Ideal): Correct + <c_math> (u_math takes precedence if both present)
    if is_correct and has_c_math and not has_u_math:
        return 2.0
    
    # C2 (Cautious): Correct + <u_math>
    if is_correct and has_u_math:
        return 1.0
    
    # C3 (Honest): Wrong + <u_math>
    if not is_correct and has_u_math:
        return 0.1
    
    # C4 (Hallucination): Wrong + <c_math>
    if not is_correct and has_c_math and not has_u_math:
        return -1.0
    
    # Default fallback (shouldn't reach here)
    return -0.5


def test_reward_function():
    """Test the reward function with various scenarios."""
    
    print("="*80)
    print("REWARD FUNCTION TESTS")
    print("="*80)
    
    # Test cases
    test_cases = [
        # C1: Confident and correct (IDEAL)
        {
            "name": "C1 (Ideal): Confident + Correct",
            "generated": "Let me solve this step by step.\n2 + 2 = 4\nSo the answer is \\boxed{4}.<c_math>",
            "ground_truth": "The solution is \\boxed{4}",
            "expected_reward": 2.0,
        },
        # C2: Cautious but correct
        {
            "name": "C2 (Cautious): Correct + Uncertain",
            "generated": "I think the answer is \\boxed{4}, but I'm not completely sure.<u_math>",
            "ground_truth": "The answer is \\boxed{4}",
            "expected_reward": 1.0,
        },
        # C3: Honest about being wrong
        {
            "name": "C3 (Honest): Wrong + Uncertain",
            "generated": "This problem is complex. My attempt: \\boxed{5}.<u_math>",
            "ground_truth": "The answer is \\boxed{4}",
            "expected_reward": 0.1,
        },
        # C4: Hallucination - confident but wrong (WORST)
        {
            "name": "C4 (Hallucination): Confident + Wrong",
            "generated": "I'm certain the answer is \\boxed{5}.<c_math>",
            "ground_truth": "The answer is \\boxed{4}",
            "expected_reward": -1.0,
        },
        # C5: Format error - no confidence tag
        {
            "name": "C5 (Format Error): No Tag",
            "generated": "The answer is \\boxed{10}",
            "ground_truth": "The answer is \\boxed{10}",
            "expected_reward": -0.5,
        },
        # C5: Format error - no answer box
        {
            "name": "C5 (Format Error): No Answer Box",
            "generated": "I'm confident this is correct!<c_math>",
            "ground_truth": "The answer is \\boxed{7}",
            "expected_reward": -0.5,
        },
        # Mathematical equivalence (fraction vs decimal) - C1
        {
            "name": "C1: Confident + Mathematically Equivalent",
            "generated": "The answer is \\boxed{0.5}.<c_math>",
            "ground_truth": "The answer is \\boxed{\\frac{1}{2}}",
            "expected_reward": 2.0,
        },
        # Mathematical equivalence (simplified form) - C1
        {
            "name": "C1: Confident + Simplified Form",
            "generated": "Therefore, \\boxed{6}.<c_math>",
            "ground_truth": "The answer is \\boxed{2 \\times 3}",
            "expected_reward": 2.0,
        },
        # Both tags present (u_math takes precedence) - correct
        {
            "name": "C2: Both Tags (u_math wins) - Correct",
            "generated": "Maybe \\boxed{5}?<c_math> Actually I'm not sure.<u_math>",
            "ground_truth": "The answer is \\boxed{5}",
            "expected_reward": 1.0,
        },
        # Both tags present (u_math takes precedence) - wrong
        {
            "name": "C3: Both Tags (u_math wins) - Wrong",
            "generated": "I think \\boxed{10}.<c_math> Wait, I'm uncertain.<u_math>",
            "ground_truth": "The answer is \\boxed{5}",
            "expected_reward": 0.1,
        },
        # Complex expression - C1
        {
            "name": "C1: Confident + Complex Expression",
            "generated": "After simplification: \\boxed{x^2 + 2x + 1}.<c_math>",
            "ground_truth": "The answer is \\boxed{(x+1)^2}",
            "expected_reward": 2.0,
        },
        # Negative numbers - C1
        {
            "name": "C1: Confident + Negative Number",
            "generated": "The result is \\boxed{-15}.<c_math>",
            "ground_truth": "Answer: \\boxed{-15}",
            "expected_reward": 2.0,
        },
    ]
    
    # Run tests
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['name']}")
        print(f"{'─'*80}")
        
        actual_reward = reward_fn(test['generated'], test['ground_truth'])
        expected_reward = test['expected_reward']
        
        status = "✓ PASS" if actual_reward == expected_reward else "✗ FAIL"
        if actual_reward == expected_reward:
            passed += 1
        else:
            failed += 1
        
        print(f"Generated: {test['generated'][:100]}...")
        print(f"Expected Reward: {expected_reward:+.1f}")
        print(f"Actual Reward:   {actual_reward:+.1f}")
        print(f"Status: {status}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total Tests: {len(test_cases)}")
    print(f"Passed: {passed} ✓")
    print(f"Failed: {failed} ✗")
    print(f"Success Rate: {100 * passed / len(test_cases):.1f}%")
    print(f"{'='*80}")
    
    return passed == len(test_cases)


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    
    print("\n" + "="*80)
    print("EDGE CASE TESTS")
    print("="*80)
    
    edge_cases = [
        {
            "name": "Empty generated text",
            "generated": "",
            "ground_truth": "\\boxed{5}",
        },
        {
            "name": "Only whitespace",
            "generated": "   \n\n  ",
            "ground_truth": "\\boxed{5}",
        },
        {
            "name": "Multiple boxes (first one used)",
            "generated": "First \\boxed{10}, wait no, \\boxed{20}.<c_math>",
            "ground_truth": "\\boxed{10}",
        },
        {
            "name": "Case sensitivity of tags",
            "generated": "Answer: \\boxed{5}.<c_math>",
            "ground_truth": "\\boxed{5}",
        },
        {
            "name": "Tag with extra spaces",
            "generated": "Answer: \\boxed{5}.< |c_math| >",
            "ground_truth": "\\boxed{5}",
        },
    ]
    
    for i, test in enumerate(edge_cases, 1):
        print(f"\nEdge Case {i}: {test['name']}")
        print(f"{'─'*80}")
        
        try:
            reward = reward_fn(test['generated'], test['ground_truth'])
            print(f"Generated: {repr(test['generated'][:100])}")
            print(f"Reward: {reward}")
            print(f"Status: ✓ No crash")
        except Exception as e:
            print(f"Generated: {repr(test['generated'][:100])}")
            print(f"Status: ✗ Error: {e}")


def show_reward_distribution():
    """Show the reward distribution for common scenarios."""
    
    print("\n" + "="*80)
    print("REWARD DISTRIBUTION GUIDE (Based on reward.txt)")
    print("="*80)
    
    scenarios = [
        ("C1 (Ideal)", "Correct + <c_math>", 2.0, "Excellent - encourages correct and confident"),
        ("C2 (Cautious)", "Correct + <u_math>", 1.0, "Good - correct but not confident"),
        ("C3 (Honest)", "Wrong + <u_math>", 0.1, "Acceptable - knows it's wrong, avoids hallucination"),
        ("C4 (Hallucination)", "Wrong + <c_math>", -1.0, "Worst - wrong but confident, strong penalty"),
        ("C5 (Format Error)", "No tag/box", -0.5, "Penalty for not following instructions"),
    ]
    
    print("\n{:<20} {:<25} {:>8}  {}".format("Case", "Condition", "Reward", "Meaning"))
    print(f"{'─'*80}")
    for case, condition, reward, meaning in scenarios:
        print(f"{case:<20} {condition:<25} {reward:+7.1f}  {meaning}")
    
    print(f"\n{'─'*80}")
    print("Key insights:")
    print("  1. C1 gets highest reward (+2.0) - be confident when correct")
    print("  2. C2 still rewarded (+1.0) - correct is good even if uncertain")
    print("  3. C3 small reward (+0.1) - honesty about uncertainty is valued")
    print("  4. C4 strong penalty (-1.0) - hallucination is the worst outcome")
    print("  5. C5 moderate penalty (-0.5) - following format is important")
    print("="*80)


if __name__ == "__main__":
    # Run all tests
    all_passed = test_reward_function()
    test_edge_cases()
    show_reward_distribution()
    
    # Final message
    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL MAIN TESTS PASSED - Reward function is working correctly!")
    else:
        print("✗ SOME TESTS FAILED - Please review the reward function.")
    print("="*80)
