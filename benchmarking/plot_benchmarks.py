#!/usr/bin/env python3
"""
Benchmark visualization suite
Generates plots from benchmark CSV outputs
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from pathlib import Path

def plot_convergence(csv_path, output_dir):
    """Plot CV convergence for each benchmark"""
    df = pd.read_csv(csv_path)

    # Group by benchmark+params
    groups = df.groupby(['benchmark_name', 'params'])

    fig, axes = plt.subplots(len(groups), 1, figsize=(12, 4*len(groups)))
    if len(groups) == 1:
        axes = [axes]

    for idx, ((bench_name, params), group) in enumerate(groups):
        ax = axes[idx]

        # Plot CV convergence
        ax.plot(group['sample_num'], group['cv_percent'], 'b-o', linewidth=2, markersize=6)
        ax.axhline(y=5.0, color='r', linestyle='--', label='Target CV (5%)')
        ax.set_xlabel('Sample Number', fontsize=12)
        ax.set_ylabel('CV (%)', fontsize=12)
        ax.set_title(f'{bench_name} ({params}) - CV Convergence', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'convergence.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved convergence plot: {output_file}")
    plt.close()


def should_regenerate_plot(benchmark_name, results_dir, plots_dir):
    """Check if plot needs regeneration based on file modification times"""
    # Check both possible plot files
    plot_1d = os.path.join(plots_dir, f'{benchmark_name}_1d.png')
    plot_2d = os.path.join(plots_dir, f'{benchmark_name}_2d.png')

    # Find the plot file that exists (if any)
    existing_plot = None
    if os.path.exists(plot_1d):
        existing_plot = plot_1d
    elif os.path.exists(plot_2d):
        existing_plot = plot_2d

    # If no plot exists, we need to generate it
    if not existing_plot:
        return True

    plot_mtime = os.path.getmtime(existing_plot)

    # Check modification times of individual result files for this benchmark
    per_benchmark_dir = os.path.join(results_dir, 'per_benchmark')
    if not os.path.exists(per_benchmark_dir):
        return True

    # Check all CSV files that start with this benchmark name
    for filename in os.listdir(per_benchmark_dir):
        if filename.startswith(benchmark_name) and filename.endswith('.csv'):
            result_file = os.path.join(per_benchmark_dir, filename)
            if os.path.getmtime(result_file) > plot_mtime:
                return True

    return False

def plot_1d_sweep(detailed_csv, output_dir, filter_benchmarks=None):
    """Plot 1D parameter sweeps"""
    df = pd.read_csv(detailed_csv)

    # Detect sub-timings in columns
    sub_timing_names = []
    for col in df.columns:
        if col.endswith('_avg_us') and col != 'avg_us':
            timing_name = col.replace('_avg_us', '')
            sub_timing_names.append(timing_name)

    # Extract benchmarks with single parameter
    single_param_benchmarks = []
    for benchmark in df['benchmark_name'].unique():
        bench_data = df[df['benchmark_name'] == benchmark]
        # Parse params to check if single parameter
        params_list = bench_data['params'].values[0].split(',') if ',' in bench_data['params'].values[0] else [bench_data['params'].values[0]]
        if len(params_list) == 1:
            single_param_benchmarks.append(benchmark)

    if not single_param_benchmarks:
        print("No 1D benchmarks found")
        return

    for benchmark in single_param_benchmarks:
        # Skip if not in filter list
        if filter_benchmarks is not None and benchmark not in filter_benchmarks:
            continue

        bench_data = df[df['benchmark_name'] == benchmark]

        # Extract parameter name and values
        param_str = bench_data['params'].values[0]
        parts = param_str.split('=')
        if len(parts) < 2:
            continue  # Skip malformed params
        param_name = parts[0].strip()

        param_values = []
        avg_times = []
        stddev_times = []
        sub_timings_data = {name: {'avg': [], 'stddev': []} for name in sub_timing_names}

        # Collect data points
        data_points = []
        for _, row in bench_data.iterrows():
            parts = row['params'].split('=')
            if len(parts) < 2:
                continue
            val = parts[1].strip()
            try:
                param_val = float(val)
            except:
                param_val = val

            point = {
                'param': param_val,
                'avg': row['avg_us'],
                'stddev': row['stddev_us'],
                'sub_timings': {}
            }

            # Collect sub-timing data for this point
            for name in sub_timing_names:
                avg_col = f'{name}_avg_us'
                stddev_col = f'{name}_stddev_us'
                if avg_col in row and pd.notna(row[avg_col]):
                    point['sub_timings'][name] = {
                        'avg': row[avg_col],
                        'stddev': row[stddev_col] if stddev_col in row else 0
                    }
                else:
                    point['sub_timings'][name] = None

            data_points.append(point)

        # Sort by parameter value to avoid zigzag lines
        data_points.sort(key=lambda p: p['param'] if isinstance(p['param'], (int, float)) else str(p['param']))

        # Extract sorted arrays
        param_values = [p['param'] for p in data_points]
        avg_times = [p['avg'] for p in data_points]
        stddev_times = [p['stddev'] for p in data_points]

        # Extract sorted sub-timing data
        for name in sub_timing_names:
            for p in data_points:
                if p['sub_timings'][name] is not None:
                    sub_timings_data[name]['avg'].append(p['sub_timings'][name]['avg'])
                    sub_timings_data[name]['stddev'].append(p['sub_timings'][name]['stddev'])
                else:
                    sub_timings_data[name]['avg'].append(None)
                    sub_timings_data[name]['stddev'].append(None)

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot sub-timings first (so they're not hidden behind overall)
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']  # Distinct, vibrant colors
        for idx, name in enumerate(sub_timing_names):
            data = sub_timings_data[name]
            if any(v is not None for v in data['avg']):
                # Filter out None values
                valid_indices = [i for i, v in enumerate(data['avg']) if v is not None]
                valid_params = [param_values[i] for i in valid_indices]
                valid_avgs = [data['avg'][i] for i in valid_indices]
                valid_stds = [data['stddev'][i] for i in valid_indices]

                ax.errorbar(valid_params, valid_avgs, yerr=valid_stds,
                           fmt='s-', linewidth=2.5, markersize=7, capsize=4, capthick=2,
                           label=name.capitalize(), color=colors[idx % len(colors)], alpha=0.9, zorder=5)

        # Check if THIS benchmark actually has sub-timing data
        has_sub_timings = any(
            any(v is not None for v in sub_timings_data[name]['avg'])
            for name in sub_timing_names
        )

        # Plot overall timing (only show if there are sub-timings, otherwise it's the only line)
        if has_sub_timings:
            # With sub-timings: show overall as dashed line with label
            ax.errorbar(param_values, avg_times, yerr=stddev_times,
                       fmt='o--', linewidth=2, markersize=7, capsize=5, capthick=2,
                       label='Overall', color='#34495e', alpha=0.6, zorder=3)
        else:
            # Without sub-timings: just show the main data without label
            ax.errorbar(param_values, avg_times, yerr=stddev_times,
                       fmt='o-', linewidth=2, markersize=7, capsize=5, capthick=2,
                       color='#2c3e50', alpha=1.0, zorder=3)

        ax.set_xlabel(param_name, fontsize=14)
        ax.set_ylabel('Time (μs)', fontsize=14)
        ax.set_title(f'{benchmark} - Performance vs {param_name}', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Only show legend if there are multiple lines (sub-timings exist)
        if has_sub_timings:
            ax.legend(fontsize=11)

        # Set x-axis limits and ticks to start from actual minimum (avoid misleading 0)
        if all(isinstance(v, (int, float)) for v in param_values):
            min_val = min(param_values)
            max_val = max(param_values)
            margin = (max_val - min_val) * 0.05  # 5% margin
            ax.set_xlim(min_val - margin, max_val + margin)

            # Limit number of ticks if too many data points
            unique_vals = sorted(set(param_values))
            if len(unique_vals) > 10:
                # Use MaxNLocator to intelligently choose ~8 ticks
                from matplotlib.ticker import MaxNLocator
                ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
            else:
                # Show all data points as ticks
                ax.set_xticks(unique_vals)

        plt.tight_layout()
        output_file = os.path.join(output_dir, f'{benchmark}_1d.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved 1D plot: {output_file}")
        plt.close()


def plot_2d_sweep(detailed_csv, output_dir, filter_benchmarks=None):
    """Plot 2D parameter sweeps as surface plots or grouped bar charts"""
    df = pd.read_csv(detailed_csv)

    # Detect sub-timings in columns
    sub_timing_names = []
    for col in df.columns:
        if col.endswith('_avg_us') and col != 'avg_us':
            timing_name = col.replace('_avg_us', '')
            sub_timing_names.append(timing_name)

    # Extract benchmarks with two parameters
    two_param_benchmarks = []
    for benchmark in df['benchmark_name'].unique():
        bench_data = df[df['benchmark_name'] == benchmark]
        params_list = bench_data['params'].values[0].split(',')
        if len(params_list) == 2:
            two_param_benchmarks.append(benchmark)

    if not two_param_benchmarks:
        print("No 2D benchmarks found")
        return

    for benchmark in two_param_benchmarks:
        # Skip if not in filter list
        if filter_benchmarks is not None and benchmark not in filter_benchmarks:
            continue

        bench_data = df[df['benchmark_name'] == benchmark]

        # Extract parameter names
        param_str = bench_data['params'].values[0]
        param1_name = param_str.split(',')[0].split('=')[0].strip()
        param2_name = param_str.split(',')[1].split('=')[0].strip()

        param1_vals = []
        param2_vals = []
        avg_times = []
        sub_timings_data = {name: [] for name in sub_timing_names}

        for _, row in bench_data.iterrows():
            parts = row['params'].split(',')
            p1_val = parts[0].split('=')[1].strip()
            p2_val = parts[1].split('=')[1].strip()

            # Try to convert to float, otherwise keep as string
            try:
                p1 = float(p1_val)
            except ValueError:
                p1 = p1_val

            try:
                p2 = float(p2_val)
            except ValueError:
                p2 = p2_val

            param1_vals.append(p1)
            param2_vals.append(p2)
            avg_times.append(row['avg_us'])

            # Collect sub-timing data
            for name in sub_timing_names:
                avg_col = f'{name}_avg_us'
                if avg_col in row and pd.notna(row[avg_col]):
                    sub_timings_data[name].append(row[avg_col])
                else:
                    sub_timings_data[name].append(None)

        # Check if both parameters are numeric
        all_numeric = all(isinstance(v, (int, float)) for v in param1_vals + param2_vals)

        if all_numeric:
            # Create 2D grid for surface plot
            unique_p1 = sorted(set(param1_vals))
            unique_p2 = sorted(set(param2_vals))

            # Check if we have sub-timings
            has_sub_timings = any(any(v is not None for v in sub_timings_data[name]) for name in sub_timing_names)

            if has_sub_timings:
                # Create subplot for total + sub-timings
                num_plots = 1 + len(sub_timing_names)
                fig = plt.figure(figsize=(18, 6 * ((num_plots + 1) // 2)))

                # Plot overall timing
                ax = fig.add_subplot(2, (num_plots + 1) // 2, 1, projection='3d')
                Z = np.zeros((len(unique_p2), len(unique_p1)))
                for p1, p2, time in zip(param1_vals, param2_vals, avg_times):
                    i = unique_p2.index(p2)
                    j = unique_p1.index(p1)
                    Z[i, j] = time

                X, Y = np.meshgrid(unique_p1, unique_p2)
                surf = ax.plot_surface(X, Y, Z, cmap='cividis', alpha=0.8)
                ax.set_xlabel(param1_name, fontsize=10)
                ax.set_ylabel(param2_name, fontsize=10)
                ax.set_zlabel('Time (μs)', fontsize=10)
                ax.set_title('Overall', fontsize=12, fontweight='bold')
                # Set axis limits to actual data range
                ax.set_xlim(min(unique_p1), max(unique_p1))
                ax.set_ylim(min(unique_p2), max(unique_p2))
                fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

                # Plot sub-timings
                for idx, name in enumerate(sub_timing_names):
                    data = sub_timings_data[name]
                    if any(v is not None for v in data):
                        ax = fig.add_subplot(2, (num_plots + 1) // 2, idx + 2, projection='3d')
                        Z_sub = np.zeros((len(unique_p2), len(unique_p1)))
                        for p1, p2, time in zip(param1_vals, param2_vals, data):
                            if time is not None:
                                i = unique_p2.index(p2)
                                j = unique_p1.index(p1)
                                Z_sub[i, j] = time

                        surf = ax.plot_surface(X, Y, Z_sub, cmap='plasma', alpha=0.8)
                        ax.set_xlabel(param1_name, fontsize=10)
                        ax.set_ylabel(param2_name, fontsize=10)
                        ax.set_zlabel('Time (μs)', fontsize=10)
                        ax.set_title(name.capitalize(), fontsize=12, fontweight='bold')
                        # Set axis limits to actual data range
                        ax.set_xlim(min(unique_p1), max(unique_p1))
                        ax.set_ylim(min(unique_p2), max(unique_p2))
                        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

                plt.suptitle(f'{benchmark} - 2D Parameter Sweep with Sub-timings', fontsize=16, fontweight='bold', y=0.98)
            else:
                # Single surface plot (no sub-timings)
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111, projection='3d')

                Z = np.zeros((len(unique_p2), len(unique_p1)))
                for p1, p2, time in zip(param1_vals, param2_vals, avg_times):
                    i = unique_p2.index(p2)
                    j = unique_p1.index(p1)
                    Z[i, j] = time

                X, Y = np.meshgrid(unique_p1, unique_p2)
                surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)

                ax.set_xlabel(param1_name, fontsize=12)
                ax.set_ylabel(param2_name, fontsize=12)
                ax.set_zlabel('Time (μs)', fontsize=12)
                ax.set_title(f'{benchmark} - 2D Parameter Sweep', fontsize=14, fontweight='bold')
                # Set axis limits to actual data range
                ax.set_xlim(min(unique_p1), max(unique_p1))
                ax.set_ylim(min(unique_p2), max(unique_p2))

                # Handle tick spacing intelligently to avoid label crowding
                from matplotlib.ticker import MaxNLocator

                # For integer parameters, use MaxNLocator with integer constraint
                if all(isinstance(v, int) or (isinstance(v, float) and v.is_integer()) for v in unique_p1):
                    # Limit to ~8 ticks max to prevent crowding
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))

                if all(isinstance(v, int) or (isinstance(v, float) and v.is_integer()) for v in unique_p2):
                    # Limit to ~8 ticks max to prevent crowding
                    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))

                fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

            plt.tight_layout()
            output_file = os.path.join(output_dir, f'{benchmark}_2d.png')
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"✓ Saved 2D plot: {output_file}")
            plt.close()
        else:
            # Create grouped bar chart for categorical data
            fig, ax = plt.subplots(figsize=(12, 6))

            # Group by first parameter
            unique_p1 = sorted(set(param1_vals), key=lambda x: (isinstance(x, str), x))
            unique_p2 = sorted(set(param2_vals), key=lambda x: (isinstance(x, str), x))

            x = np.arange(len(unique_p1))
            width = 0.8 / len(unique_p2)

            for i, p2 in enumerate(unique_p2):
                times = []
                for p1 in unique_p1:
                    for p1_val, p2_val, time in zip(param1_vals, param2_vals, avg_times):
                        if p1_val == p1 and p2_val == p2:
                            times.append(time)
                            break
                    else:
                        times.append(0)

                ax.bar(x + i * width, times, width, label=f'{param2_name}={p2}')

            ax.set_xlabel(param1_name, fontsize=14)
            ax.set_ylabel('Time (μs)', fontsize=14)
            ax.set_title(f'{benchmark} - Parameter Comparison', fontsize=16, fontweight='bold')
            ax.set_xticks(x + width * (len(unique_p2) - 1) / 2)
            ax.set_xticklabels([f'{p}' for p in unique_p1])
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

            # Set y-axis to start from 0 for bar charts (standard practice)
            ax.set_ylim(bottom=0)

            plt.tight_layout()
            output_file = os.path.join(output_dir, f'{benchmark}_2d.png')
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"✓ Saved 2D plot: {output_file}")
            plt.close()


def plot_time_series(benchmark_results_dir, output_dir):
    """Plot performance over time (across commits)"""
    # Collect all summary.csv files
    commit_dirs = [d for d in Path(benchmark_results_dir).iterdir() if d.is_dir()]

    if len(commit_dirs) < 2:
        print("Need at least 2 commits for time-series plot")
        return

    # Aggregate data
    all_data = []
    for commit_dir in sorted(commit_dirs):
        summary_file = commit_dir / 'summary.csv'
        if summary_file.exists():
            df = pd.read_csv(summary_file)
            all_data.append(df)

    if not all_data:
        print("No summary.csv files found")
        return

    # Combine all summaries
    combined = pd.concat(all_data, ignore_index=True)

    # Convert commit_date to datetime
    combined['commit_date'] = pd.to_datetime(combined['commit_date'])

    # Extract benchmark columns (exclude commit_hash, commit_date, timestamp)
    meta_cols = ['commit_hash', 'commit_date', 'timestamp']
    benchmark_cols = [col for col in combined.columns if col not in meta_cols and col.endswith('_avg_us')]

    # Plot each benchmark over time
    for bench_col in benchmark_cols:
        bench_name = bench_col.replace('_avg_us', '')

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(combined['commit_date'], combined[bench_col], 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Commit Date', fontsize=14)
        ax.set_ylabel('Time (μs)', fontsize=14)
        ax.set_title(f'{bench_name} - Performance Over Time', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        output_file = os.path.join(output_dir, f'{bench_name}_timeseries.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved time-series plot: {output_file}")
        plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_benchmarks.py <benchmark_results_dir>")
        print("Example: python plot_benchmarks.py benchmark_results/a05e49a1...")
        sys.exit(1)

    results_dir = sys.argv[1]

    if not os.path.exists(results_dir):
        print(f"Error: Directory {results_dir} does not exist")
        sys.exit(1)

    # Create plots directory
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    print(f"Generating plots from {results_dir}...")
    print(f"Output directory: {plots_dir}\n")

    # Generate all plot types
    # Convergence plot disabled - only needed for tuning adaptive sampling
    # convergence_csv = os.path.join(results_dir, 'samples_convergence.csv')
    # if os.path.exists(convergence_csv):
    #     plot_convergence(convergence_csv, plots_dir)

    detailed_csv = os.path.join(results_dir, 'detailed.csv')

    if os.path.exists(detailed_csv):
        # Load benchmark data to get list of benchmarks
        df = pd.read_csv(detailed_csv)
        benchmarks_to_plot = []
        new_plots = []
        updated_plots = []

        for benchmark in df['benchmark_name'].unique():
            # Check if plot exists
            plot_1d = os.path.join(plots_dir, f'{benchmark}_1d.png')
            plot_2d = os.path.join(plots_dir, f'{benchmark}_2d.png')
            plot_exists = os.path.exists(plot_1d) or os.path.exists(plot_2d)

            if should_regenerate_plot(benchmark, results_dir, plots_dir):
                benchmarks_to_plot.append(benchmark)
                if plot_exists:
                    updated_plots.append(benchmark)
                    print(f"  → Will regenerate plot for: {benchmark}")
                else:
                    new_plots.append(benchmark)
                    print(f"  → Will generate plot for: {benchmark}")
            else:
                print(f"  ✓ Plot up-to-date for: {benchmark}")

        if benchmarks_to_plot:
            if new_plots and updated_plots:
                print(f"\nGenerating {len(new_plots)} new plot(s) and regenerating {len(updated_plots)} plot(s)...\n")
            elif new_plots:
                print(f"\nGenerating {len(new_plots)} plot(s)...\n")
            else:
                print(f"\nRegenerating {len(updated_plots)} plot(s)...\n")

            # Generate plots only for changed benchmarks
            plot_1d_sweep(detailed_csv, plots_dir, filter_benchmarks=benchmarks_to_plot)
            plot_2d_sweep(detailed_csv, plots_dir, filter_benchmarks=benchmarks_to_plot)
        else:
            print("\nAll plots are up-to-date!\n")

    # Time-series requires parent directory with multiple commits
    parent_dir = os.path.dirname(results_dir)
    if os.path.basename(parent_dir) == 'benchmark_results':
        plot_time_series(parent_dir, plots_dir)

    # Always generate BENCHMARKS.md (needs full results table)
    generate_benchmarks_md(results_dir, plots_dir, detailed_csv)

    print(f"\n✓ All plots saved to: {plots_dir}")
    print(f"✓ BENCHMARKS.md generated: {os.path.join(results_dir, 'BENCHMARKS.md')}")


def get_hardware_info():
    """Collect system hardware information"""
    import platform
    import subprocess

    info = {}

    # CPU information
    try:
        if platform.system() == "Linux":
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                for line in cpuinfo.split('\n'):
                    if 'model name' in line:
                        info['cpu'] = line.split(':')[1].strip()
                        break

            # Count cores and threads
            info['cores'] = int(subprocess.check_output(['nproc', '--all']).decode().strip())
        else:
            info['cpu'] = platform.processor()
            info['cores'] = os.cpu_count()
    except:
        info['cpu'] = "Unknown"
        info['cores'] = os.cpu_count() or "Unknown"

    # Architecture
    info['arch'] = platform.machine()

    # OS
    info['os'] = f"{platform.system()} {platform.release()}"

    # Memory
    try:
        if platform.system() == "Linux":
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
                for line in meminfo.split('\n'):
                    if 'MemTotal' in line:
                        mem_kb = int(line.split(':')[1].strip().split()[0])
                        info['memory'] = f"{mem_kb // 1024 // 1024} GB"
                        break
        else:
            info['memory'] = "Unknown"
    except:
        info['memory'] = "Unknown"

    # Compiler version - will be read from config file, not queried
    info['compiler'] = "Unknown"

    return info


def get_benchmark_metadata():
    """Extract benchmark metadata (category, descriptions) from all cpp files in benchmarks folder"""
    import re
    import glob

    metadata = {}  # benchmark_name -> {'description': ..., 'category': ...}

    # Read all .cpp files in benchmarks directory
    benchmarks_dir = os.path.join(os.path.dirname(__file__), '..', 'benchmarks')
    for benchmark_file in glob.glob(os.path.join(benchmarks_dir, '*.cpp')):
        with open(benchmark_file, 'r') as f:
            content = f.read()

            # Extract category from comment: // BENCHMARK_CATEGORY: Category Name
            category_match = re.search(r'//\s*BENCHMARK_CATEGORY:\s*(.+)', content)
            file_category = category_match.group(1).strip() if category_match else None

            # Match BENCHMARK(name, "description") pattern
            pattern = r'BENCHMARK\((\w+),\s*"([^"]+)"\)'
            matches = re.findall(pattern, content)
            for name, desc in matches:
                metadata[name] = {
                    'description': desc,
                    'category': file_category
                }

    return metadata


def generate_benchmarks_md(results_dir, plots_dir, detailed_csv):
    """Generate BENCHMARKS.md with hardware info and plots"""
    config_file = os.path.join(results_dir, 'benchmark_config.txt')

    # Read config
    config = {}
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    config[key] = value

    # Get hardware info
    hw = get_hardware_info()

    # Override compiler from config if available
    if 'COMPILER' in config:
        hw['compiler'] = config['COMPILER']

    # Get benchmark metadata (descriptions and categories)
    metadata = get_benchmark_metadata()

    # Read benchmark data
    df = pd.read_csv(detailed_csv)

    # Group benchmarks by category from metadata
    categories = {}
    uncategorized = []

    for benchmark in df['benchmark_name'].unique():
        if benchmark in metadata and metadata[benchmark]['category']:
            category = metadata[benchmark]['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(benchmark)
        else:
            uncategorized.append(benchmark)

    # Add uncategorized benchmarks to "Other" category if any exist
    if uncategorized:
        categories['Other'] = uncategorized

    # Generate markdown
    md = []
    md.append("# Hypergraph Engine Benchmarks\n")

    # Table of Contents
    md.append("## Contents\n")
    for category, benchmarks in categories.items():
        if not benchmarks:
            continue
        category_anchor = category.lower().replace(' ', '-') + '-benchmarks'
        md.append(f"- **[{category} Benchmarks](#{category_anchor})**")
        for benchmark in sorted(benchmarks):
            md.append(f"  - [{benchmark}](#{benchmark})")
    md.append("")

    # System information
    md.append("## System Information\n")
    md.append(f"- **CPU**: {hw['cpu']}")
    md.append(f"- **Cores**: {hw['cores']}")
    md.append(f"- **Architecture**: {hw['arch']}")
    md.append(f"- **OS**: {hw['os']}")
    md.append(f"- **Memory**: {hw['memory']}")
    md.append(f"- **Compiler**: {hw['compiler']}")
    md.append(f"- **Hash Type**: {config.get('HASH_TYPE', 'unknown')}")
    md.append(f"- **Hash**: {config.get('HASH', 'unknown')}")
    md.append(f"- **Date**: {config.get('COMMIT_DATE', 'unknown')}")
    md.append(f"- **Timestamp**: {config.get('TIMESTAMP', 'unknown')}\n")

    # Benchmarks by category
    for category, benchmarks in categories.items():
        if not benchmarks:
            continue

        md.append(f"## {category} Benchmarks\n")

        for benchmark in sorted(benchmarks):
            md.append(f"### {benchmark}\n")

            # Add description if available
            if benchmark in metadata and metadata[benchmark]['description']:
                md.append(f"{metadata[benchmark]['description']}\n")

            # Check for plot files
            plot_1d = os.path.join('plots', f'{benchmark}_1d.png')
            plot_2d = os.path.join('plots', f'{benchmark}_2d.png')

            if os.path.exists(os.path.join(results_dir, plot_1d)):
                md.append(f"![{benchmark}]({plot_1d})\n")
            elif os.path.exists(os.path.join(results_dir, plot_2d)):
                md.append(f"![{benchmark}]({plot_2d})\n")

            # Add metrics table
            bench_data = df[df['benchmark_name'] == benchmark]
            if len(bench_data) > 0:
                # Detect sub-timings for this benchmark
                sub_timing_cols = []
                for col in bench_data.columns:
                    if col.endswith('_avg_us') and col != 'avg_us':
                        # Check if this benchmark actually has data for this sub-timing
                        if bench_data[col].notna().any():
                            timing_name = col.replace('_avg_us', '')
                            sub_timing_cols.append(timing_name)

                # Build table header
                if sub_timing_cols:
                    # Has sub-timings: show Overall + sub-timings
                    header = "| Parameters | Overall (μs) | Stddev (μs) | CV% | Samples |"
                    separator = "|------------|--------------|-------------|-----|---------|"
                    for name in sub_timing_cols:
                        header += f" {name.capitalize()} (μs) |"
                        separator += "-----------------|"
                else:
                    # No sub-timings: show Avg instead of Overall
                    header = "| Parameters | Avg (μs) | Stddev (μs) | CV% | Samples |"
                    separator = "|------------|----------|-------------|-----|---------|"

                md.append(header)
                md.append(separator)

                # Build table rows
                for _, row in bench_data.iterrows():
                    params = row['params']
                    row_str = f"| {params} | {row['avg_us']:.2f} | {row['stddev_us']:.2f} | {row['cv_percent']:.2f} | {row['samples']} |"
                    for name in sub_timing_cols:
                        avg_col = f'{name}_avg_us'
                        if pd.notna(row[avg_col]):
                            row_str += f" {row[avg_col]:.2f} |"
                        else:
                            row_str += " - |"
                    md.append(row_str)
                md.append("")

    # Write file
    output_file = os.path.join(results_dir, 'BENCHMARKS.md')
    with open(output_file, 'w') as f:
        f.write('\n'.join(md))


if __name__ == '__main__':
    main()
