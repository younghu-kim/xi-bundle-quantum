#!/usr/bin/env python
"""
양자 컴퓨팅 모듈 통합 실행 스크립트

5개 모듈을 순서대로 실행하고 결과를 종합:
1. DQPT 시뮬레이션 (dqpt/dqpt_zeta.py)
2. Floquet 영점 탐지 (dqpt/floquet_zeros.py)
3. 해밀토니안 제약 조건 검증 (hamiltonian/constraints.py)
4. Bethe-GB 등가성 (hamiltonian/bethe_ansatz_gb.py)
5. VQE + GB 제약 (vqe/vqe_gauss_bonnet.py)
6. ξ-다발 위상 코드 (topological/xi_bundle_code.py)

결과: ~/Desktop/gdl_unified/results/quantum_summary.txt
"""
import sys
import os
import time
import traceback

QUANTUM_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(QUANTUM_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# 경로 설정
sys.path.insert(0, QUANTUM_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, 'scripts'))


def run_module(name, module_path):
    """모듈 실행 + 시간 측정"""
    print(f"\n{'='*70}")
    print(f"  [{name}] 실행 시작")
    print(f"{'='*70}\n")

    start = time.time()
    try:
        # 모듈을 exec로 실행
        with open(module_path, 'r') as f:
            code = f.read()
        exec(compile(code, module_path, 'exec'), {'__name__': '__main__'})
        elapsed = time.time() - start
        print(f"\n  ✅ [{name}] 완료 ({elapsed:.1f}초)")
        return True, elapsed
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n  ❌ [{name}] 실패 ({elapsed:.1f}초): {e}")
        traceback.print_exc()
        return False, elapsed


def main():
    print("=" * 70)
    print("  ξ-다발 양자 컴퓨팅 모듈 통합 실행")
    print(f"  시작 시각: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    modules = [
        ("Bethe-GB 등가성", os.path.join(QUANTUM_DIR, "hamiltonian", "bethe_ansatz_gb.py")),
        ("Floquet 영점 탐지", os.path.join(QUANTUM_DIR, "dqpt", "floquet_zeros.py")),
        ("DQPT 시뮬레이션", os.path.join(QUANTUM_DIR, "dqpt", "dqpt_zeta.py")),
        ("해밀토니안 제약 검증", os.path.join(QUANTUM_DIR, "hamiltonian", "constraints.py")),
        ("VQE-GB 최적화", os.path.join(QUANTUM_DIR, "vqe", "vqe_gauss_bonnet.py")),
        ("ξ-다발 위상 코드", os.path.join(QUANTUM_DIR, "topological", "xi_bundle_code.py")),
    ]

    results = []
    for name, path in modules:
        if os.path.exists(path):
            success, elapsed = run_module(name, path)
            results.append((name, success, elapsed))
        else:
            print(f"\n  ⏭️ [{name}] 파일 없음: {path}")
            results.append((name, None, 0))

    # 요약
    print(f"\n\n{'='*70}")
    print("  통합 실행 결과 요약")
    print(f"{'='*70}\n")

    total_time = sum(r[2] for r in results)
    n_success = sum(1 for r in results if r[1] is True)
    n_fail = sum(1 for r in results if r[1] is False)
    n_skip = sum(1 for r in results if r[1] is None)

    for name, success, elapsed in results:
        status = "✅ 성공" if success is True else "❌ 실패" if success is False else "⏭️ 스킵"
        print(f"  {status}  {name:<25s}  ({elapsed:.1f}초)")

    print(f"\n  합계: {n_success} 성공 / {n_fail} 실패 / {n_skip} 스킵")
    print(f"  총 소요: {total_time:.1f}초")

    # 결과 파일 저장
    summary_path = os.path.join(RESULTS_DIR, 'quantum_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"ξ-다발 양자 컴퓨팅 모듈 통합 실행 결과\n")
        f.write(f"실행 시각: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*50}\n\n")
        for name, success, elapsed in results:
            status = "성공" if success is True else "실패" if success is False else "스킵"
            f.write(f"[{status}] {name} ({elapsed:.1f}초)\n")
        f.write(f"\n합계: {n_success}/{len(results)} 성공, {total_time:.1f}초\n")

    print(f"\n  결과 저장: {summary_path}")


if __name__ == '__main__':
    main()
