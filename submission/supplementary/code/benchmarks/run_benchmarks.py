#!/usr/bin/env python3
"""Run performance benchmarks."""

import json
import argparse
from datetime import datetime
from pathlib import Path
from benchmark import PerformanceBenchmark

def main():
    parser = argparse.ArgumentParser(description='Run performance benchmarks')
    parser.add_argument('--output', type=str, default='benchmark_results',
                       help='Output directory for results')
    args = parser.parse_args()
    
    # Run benchmarks
    benchmark = PerformanceBenchmark()
    results = benchmark.run_all_benchmarks()
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'benchmark_results_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump([{
            'name': r.name,
            'execution_time': r.execution_time,
            'memory_usage': r.memory_usage,
            'throughput': r.throughput,
            'parameters': r.parameters
        } for r in results], f, indent=2)
    
    print(f'Benchmark results saved to {output_file}')

if __name__ == '__main__':
    main() 