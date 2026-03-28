import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import warnings
import time
from datetime import datetime
import os
from scipy import stats
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - CHỈ 3 MODELS
# ============================================================================

class Config:
    """Cấu hình cho benchmark comparison - Chỉ 3 models dùng dữ liệu tương tự"""
    
    BASE_PATH = "bruteforce_csa_full_scale"
    YOUR_MODEL_NAME = "FraGraphDRP"
    
    # Experiments (Source-Target format)
    EXPERIMENTS = ["CCLE-CCLE", "gCSI-gCSI", "CCLE-gCSI", "gCSI-CCLE"]
    
    # Metrics
    METRICS = ['mse', 'rmse', 'pcc', 'scc', 'r2']
    PRIMARY_METRIC = 'r2'
    
    # ✅ CHỈ 3 MODELS DÙNG DỮ LIỆU TƯƠNG TỰ
    BENCHMARK_MODELS = ['DeepCDR', 'GraphDRP', 'tCNNs']
    
    # Colors
    BENCHMARK_COLORS = {
        'DeepCDR': '#1f77b4',    # Xanh dương
        'GraphDRP': '#ff7f0e',   # Cam
        'tCNNs': '#2ca02c'       # Xanh lá
    }
    
    YOUR_MODEL_COLOR = '#d62728'  # Đỏ nổi bật cho FraGraphDRP
    
    DPI = 200
    
    # Output directories
    OUTPUT_DIR = "benchmark_analysis_3models"
    PLOTS_DIR = f"{OUTPUT_DIR}/plots"
    REPORTS_DIR = f"{OUTPUT_DIR}/reports"
    DATA_DIR = f"{OUTPUT_DIR}/data"


# ============================================================================
# BENCHMARK DATA - CHỈ 3 MODELS (Từ Partin et al. 2025)
# ============================================================================

class BenchmarkData:
    """
    Dữ liệu benchmark từ paper Partin et al. 2025
    Chỉ bao gồm 3 models: DeepCDR, GraphDRP, tCNNs
    Source: Tables 4, 5 và Figure 5 trong paper
    """
    
    # Table 4: Within-dataset R² mean scores - CHỈ 3 MODELS
    WITHIN_DATASET_R2_MEAN = {
        'CCLE': {
            'DeepCDR': 0.766,
            'GraphDRP': 0.746,
            'tCNNs': 0.705
        },
        'gCSI': {
            'DeepCDR': 0.720,
            'GraphDRP': 0.736,
            'tCNNs': 0.591
        },
        'GDSCv2': {
            'DeepCDR': 0.760,
            'GraphDRP': 0.765,
            'tCNNs': 0.648
        },
        'GDSCv1': {
            'DeepCDR': 0.704,
            'GraphDRP': 0.733,
            'tCNNs': 0.575
        },
        'CTRPv2': {
            'DeepCDR': 0.811,
            'GraphDRP': 0.855,
            'tCNNs': 0.639
        }
    }
    
    # Table 5: Within-dataset R² std scores - CHỈ 3 MODELS
    WITHIN_DATASET_R2_STD = {
        'CCLE': {
            'DeepCDR': 0.023,
            'GraphDRP': 0.018,
            'tCNNs': 0.049
        },
        'gCSI': {
            'DeepCDR': 0.020,
            'GraphDRP': 0.029,
            'tCNNs': 0.061
        },
        'GDSCv2': {
            'DeepCDR': 0.007,
            'GraphDRP': 0.008,
            'tCNNs': 0.052
        },
        'GDSCv1': {
            'DeepCDR': 0.008,
            'GraphDRP': 0.007,
            'tCNNs': 0.049
        },
        'CTRPv2': {
            'DeepCDR': 0.005,
            'GraphDRP': 0.006,
            'tCNNs': 0.063
        }
    }
    
    # Figure 5: Cross-dataset R² scores (approximate from heatmaps)
    CROSS_DATASET_R2 = {
        'CCLE-gCSI': {
            'DeepCDR': 0.120,
            'GraphDRP': 0.080,
            'tCNNs': -0.050
        },
        'gCSI-CCLE': {
            'DeepCDR': 0.180,
            'GraphDRP': 0.150,
            'tCNNs': 0.050
        }
    }
    
    # Best performers trong 3 models này
    BEST_PERFORMERS = {
        'CCLE': {'model': 'DeepCDR', 'r2': 0.766},
        'gCSI': {'model': 'GraphDRP', 'r2': 0.736},
        'CTRPv2': {'model': 'GraphDRP', 'r2': 0.855}
    }
    
    # Architecture info từ Table 2
    ARCHITECTURE_INFO = {
        'DeepCDR': {
            'year': 2020,
            'framework': 'TF-Keras',
            'omics': 'GE, Methyl, Mu',
            'drug': 'Molecular Graph',
            'components': 'Batchnorm, Dropout'
        },
        'GraphDRP': {
            'year': 2022,
            'framework': 'PyTorch',
            'omics': 'CNV, Mu',
            'drug': 'Molecular Graph',
            'components': 'Batchnorm, Dropout, GIN, 1D-CNN'
        },
        'tCNNs': {
            'year': 2019,
            'framework': 'TF',
            'omics': 'CNV, Mu',
            'drug': 'SMILES (one-hot)',
            'components': '1D-CNN, Dropout'
        }
    }


# ============================================================================
# YOUR RESULTS ANALYZER
# ============================================================================

class FraGraphDRPAnalyzer:
    """Phân tích kết quả của FraGraphDRP"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.metrics = Config.METRICS
    
    def load_split_result(self, experiment: str, split: int) -> Optional[Dict]:
        """Load kết quả từ file test_scores.json"""
        file_path = self.base_path / "infer" / experiment / f"split_{split}" / "test_scores.json"
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return None
    
    def analyze_experiment(self, experiment: str) -> Dict:
        """Phân tích một experiment với 5 folds"""
        splits_data = {}
        
        for fold in range(5):
            data = self.load_split_result(experiment, fold)
            if data is not None:
                splits_data[f'split_{fold}'] = data
        
        if not splits_data:
            return {}
        
        stats = {
            'experiment': experiment,
            'num_splits': len(splits_data),
            'splits_found': list(splits_data.keys())
        }
        
        for metric in self.metrics:
            values = [data.get(metric) for data in splits_data.values() 
                     if data.get(metric) is not None]
            if values:
                stats[f'{metric}_mean'] = np.mean(values)
                stats[f'{metric}_std'] = np.std(values)
                stats[f'{metric}_min'] = np.min(values)
                stats[f'{metric}_max'] = np.max(values)
                stats[f'{metric}_median'] = np.median(values)
                stats[f'{metric}_ci95'] = stats[f'{metric}_std'] * 1.96 / np.sqrt(len(values))
        
        return stats
    
    def analyze_all(self, experiments: List[str]) -> pd.DataFrame:
        """Phân tích tất cả experiments"""
        print("\n🔍 Loading FraGraphDRP results...")
        print("-" * 80)
        
        all_stats = []
        
        for exp in experiments:
            stats = self.analyze_experiment(exp)
            
            if stats and stats.get('num_splits', 0) > 0:
                all_stats.append(stats)
                r2_val = stats.get('r2_mean', 0)
                r2_std = stats.get('r2_std', 0)
                print(f"  ✓ {exp:15s}: R² = {r2_val:.4f} ± {r2_std:.4f} ({stats['num_splits']}/5 splits)")
            else:
                print(f"  ❌ {exp:15s}: No data found")
        
        print("-" * 80)
        
        return pd.DataFrame(all_stats) if all_stats else pd.DataFrame()


# ============================================================================
# COMPARATOR CLASS - CHỈ 3 MODELS
# ============================================================================

class FraGraphDRPComparator:
    """So sánh FraGraphDRP với GraphDRP, tCNNs, DeepCDR"""
    
    def __init__(self, your_results_df: pd.DataFrame, benchmark: BenchmarkData):
        self.your_df = your_results_df
        self.benchmark = benchmark
        self.your_model_name = Config.YOUR_MODEL_NAME
        
        sns.set_style("whitegrid")
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'figure.titlesize': 13
        })
    
    def extract_dataset_from_experiment(self, experiment: str) -> str:
        """Extract dataset name từ experiment"""
        if '-' in experiment:
            parts = experiment.split('-')
            if parts[0] == parts[1]:
                return parts[0]
            else:
                return experiment
        return experiment
    
    def create_comparison_df(self, dataset: str, comparison_type: str = 'within') -> pd.DataFrame:
        """Tạo DataFrame so sánh - CHỈ 3 MODELS"""
        comparison_data = []
        
        # Thêm 3 benchmark models
        if comparison_type == 'within':
            benchmark_data = self.benchmark.WITHIN_DATASET_R2_MEAN.get(dataset, {})
            benchmark_std = self.benchmark.WITHIN_DATASET_R2_STD.get(dataset, {})
        else:
            benchmark_data = self.benchmark.CROSS_DATASET_R2.get(dataset, {})
            benchmark_std = {}
        
        for model, r2 in benchmark_data.items():
            std = benchmark_std.get(model, 0.02)
            comparison_data.append({
                'Model': model,
                'R2_Mean': r2,
                'R2_Std': std,
                'Source': 'Benchmark (Partin et al. 2025)',
                'Dataset': dataset,
                'Type': comparison_type
            })
        
        # Thêm FraGraphDRP
        if not self.your_df.empty:
            for _, row in self.your_df.iterrows():
                exp_name = row['experiment']
                
                if comparison_type == 'within':
                    if '-' in exp_name:
                        parts = exp_name.split('-')
                        if parts[0] == parts[1] == dataset:
                            comparison_data.append({
                                'Model': self.your_model_name,
                                'R2_Mean': row['r2_mean'],
                                'R2_Std': row.get('r2_std', 0),
                                'Source': 'FraGraphDRP',
                                'Dataset': dataset,
                                'Type': comparison_type
                            })
                            break
                else:
                    if exp_name == dataset:
                        comparison_data.append({
                            'Model': self.your_model_name,
                            'R2_Mean': row['r2_mean'],
                            'R2_Std': row.get('r2_std', 0),
                            'Source': 'FraGraphDRP',
                            'Dataset': dataset,
                            'Type': comparison_type
                        })
                        break
        
        return pd.DataFrame(comparison_data)
    
    def plot_comparison_bar(self, dataset: str, comparison_type: str = 'within',
                           save_path: str = None) -> Optional[plt.Figure]:
        """Vẽ bar plot so sánh - CHỈ 3 MODELS"""
        df = self.create_comparison_df(dataset, comparison_type)
        
        if df.empty:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=Config.DPI)
        
        benchmark = df[df['Source'] == 'Benchmark (Partin et al. 2025)']
        yours = df[df['Source'] == 'FraGraphDRP']
        
        # Vẽ 3 benchmark models
        models = benchmark['Model'].values
        r2_scores = benchmark['R2_Mean'].values
        stds = benchmark['R2_Std'].values
        
        colors = [Config.BENCHMARK_COLORS.get(m, '#999999') for m in models]
        ax.bar(range(len(models)), r2_scores, yerr=stds, capsize=6,
              color=colors, alpha=0.75, edgecolor='black', linewidth=1.0)
        
        # Vẽ FraGraphDRP
        if not yours.empty:
            your_r2 = yours['R2_Mean'].values[0]
            your_std = yours['R2_Std'].values[0]
            
            bar_pos = len(models)
            ax.bar(bar_pos, your_r2, yerr=your_std, capsize=6,
                  color=Config.YOUR_MODEL_COLOR, alpha=0.9,
                  label=self.your_model_name, edgecolor='black',
                  linewidth=2.5, zorder=10)
            
            # Thêm text giá trị
            ax.text(bar_pos, your_r2 + 0.035, f'{your_r2:.3f}±{your_std:.3f}',
                   ha='center', va='bottom', fontweight='bold', color='darkred',
                   fontsize=11, bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                          edgecolor='darkred', alpha=0.95))
            
            # So sánh với best benchmark
            if not benchmark.empty:
                best_bench = benchmark.loc[benchmark['R2_Mean'].idxmax()]
                diff = your_r2 - best_bench['R2_Mean']
                if diff > 0:
                    status_text = f'+{diff:.3f} vs Best'
                    status_color = 'green'
                else:
                    status_text = f'{diff:.3f} vs Best'
                    status_color = 'orange'
                
                ax.text(bar_pos, your_r2 + 0.08, status_text,
                       ha='center', va='bottom', fontweight='bold', 
                       color=status_color, fontsize=9)
        
        # Customize
        all_models = list(models) + ([self.your_model_name] if not yours.empty else [])
        ax.set_xticks(range(len(all_models)))
        ax.set_xticklabels(all_models, rotation=45, ha='right', fontsize=9, fontweight='medium')
        ax.set_ylabel('R² Score', fontsize=11, fontweight='semibold')
        
        type_label = "Within-Dataset" if comparison_type == 'within' else "Cross-Dataset"
        ax.set_title(f'{type_label} Comparison\n{dataset} Dataset\n(GraphDRP, tCNNs, DeepCDR)',
                    fontsize=13, fontweight='bold', pad=15)
        ax.legend(loc='lower right', framealpha=0.95)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.set_ylim(0, 1.05)
        
        # Thêm đường reference cho best benchmark
        if dataset in self.benchmark.BEST_PERFORMERS and comparison_type == 'within':
            best_r2 = self.benchmark.BEST_PERFORMERS[dataset]['r2']
            ax.axhline(y=best_r2, color='green', linestyle='--', alpha=0.6,
                      linewidth=2.5, label=f'Best: {best_r2:.3f}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=Config.DPI, bbox_inches='tight', facecolor='white')
            print(f"  ✓ Saved: {save_path}")
        
        return fig
    
    def plot_all_comparisons(self, save_dir: str = None) -> Dict[str, str]:
        """Vẽ so sánh cho tất cả datasets"""
        if save_dir is None:
            save_dir = Config.PLOTS_DIR
        os.makedirs(save_dir, exist_ok=True)
        
        saved_files = {}
        
        print(f"\n📊 Creating comparison plots...")
        print("-" * 80)
        
        # Within-dataset
        for dataset in ['CCLE', 'gCSI']:
            path = f"{save_dir}/{dataset}_within_comparison.png"
            fig = self.plot_comparison_bar(dataset, 'within', save_path=path)
            if fig:
                saved_files[f'{dataset}_within'] = path
        
        # Cross-dataset
        for dataset in ['CCLE-gCSI', 'gCSI-CCLE']:
            path = f"{save_dir}/{dataset.replace('-', '_')}_cross_comparison.png"
            fig = self.plot_comparison_bar(dataset, 'cross', save_path=path)
            if fig:
                saved_files[f'{dataset}_cross'] = path
        
        print("-" * 80)
        print(f"  📁 Total plots saved: {len(saved_files)}")
        
        return saved_files
    
    def print_ranking(self, dataset: str, comparison_type: str = 'within'):
        """In bảng xếp hạng - CHỈ 3 MODELS"""
        df = self.create_comparison_df(dataset, comparison_type)
        
        if df.empty:
            return
        
        print("\n" + "="*90)
        print(f"📋 PERFORMANCE RANKING - {dataset} ({comparison_type})")
        print(f"   Comparing FraGraphDRP with GraphDRP, tCNNs, DeepCDR")
        print("="*90)
        
        df_sorted = df.sort_values('R2_Mean', ascending=False).reset_index(drop=True)
        df_sorted['Rank'] = range(1, len(df_sorted) + 1)
        
        benchmark = df[df['Source'] == 'Benchmark (Partin et al. 2025)']
        yours = df[df['Source'] == 'FraGraphDRP']
        
        best_benchmark = None
        if not benchmark.empty:
            best_benchmark = benchmark.loc[benchmark['R2_Mean'].idxmax()]
        
        print(f"{'Rank':<6}{'Model':<15}{'R² Mean':<12}{'R² Std':<12}{'Status':<45}")
        print("-"*90)
        
        for _, row in df_sorted.iterrows():
            if row['Source'] == 'FraGraphDRP':
                marker = "🏆"
                if best_benchmark is not None:
                    diff = row['R2_Mean'] - best_benchmark['R2_Mean']
                    if diff > 0.05:
                        status = "✅ EXCELLENT (>5% better than best)"
                    elif diff > 0:
                        status = f"✅ BETTER (+{diff:.4f} vs {best_benchmark['Model']})"
                    elif diff > -0.05:
                        status = f"⚠️  COMPETITIVE ({diff:.4f} vs {best_benchmark['Model']})"
                    else:
                        status = f"❌ BELOW BEST ({diff:.4f} vs {best_benchmark['Model']})"
                else:
                    status = ""
            else:
                marker = "  "
                status = ""
            
            print(f"{marker}#{row['Rank']:<5}{row['Model']:<15}{row['R2_Mean']:<12.4f}{row['R2_Std']:<12.4f}{status}")
        
        print("="*90)
        
        # Summary statistics
        if not yours.empty and not benchmark.empty:
            your_r2 = yours['R2_Mean'].values[0]
            your_std = yours['R2_Std'].values[0]
            bench_mean = benchmark['R2_Mean'].mean()
            bench_best = benchmark['R2_Mean'].max()
            
            percentile = (benchmark['R2_Mean'] < your_r2).sum() / len(benchmark) * 100
            
            print(f"\n📈 SUMMARY STATISTICS:")
            print(f"   {'FraGraphDRP R²:':<25} {your_r2:.4f} ± {your_std:.4f}")
            print(f"   {'Benchmark Average:':<25} {bench_mean:.4f}")
            print(f"   {'Benchmark Best:':<25} {bench_best:.4f} ({best_benchmark['Model']})")
            print(f"\n   {'vs Average:':<25} {'+' if your_r2 > bench_mean else ''}{(your_r2 - bench_mean):.4f} ({(your_r2/bench_mean - 1)*100:+.2f}%)")
            print(f"   {'vs Best:':<25} {'+' if your_r2 > bench_best else ''}{(your_r2 - bench_best):.4f} ({(your_r2/bench_best - 1)*100:+.2f}%)")
            print(f"   {'Percentile Rank:':<25} {percentile:.1f}% (outperforms {percentile:.1f}% of 3 benchmarks)")
            
            # Đánh giá
            print(f"\n🎯 EVALUATION:")
            if your_r2 >= bench_best:
                print(f"   ⭐ NEW STATE-OF-THE-ART for {dataset}!")
                print(f"   📝 FraGraphDRP outperforms GraphDRP, tCNNs, and DeepCDR")
            elif your_r2 >= bench_mean:
                print(f"   ✅ ABOVE AVERAGE - Competitive with top models")
            else:
                print(f"   ⚠️  BELOW AVERAGE - Room for improvement")
        
        print("="*90)
    
    def print_architecture_comparison(self):
        """In so sánh kiến trúc"""
        print("\n" + "="*90)
        print("📐 ARCHITECTURE COMPARISON")
        print("="*90)
        print(f"{'Model':<15}{'Year':<8}{'Framework':<12}{'Omics':<20}{'Drug':<20}{'Components':<25}")
        print("-"*90)
        
        for model in ['DeepCDR', 'GraphDRP', 'tCNNs']:
            info = self.benchmark.ARCHITECTURE_INFO.get(model, {})
            print(f"{model:<15}{info.get('year', 'N/A'):<8}{info.get('framework', 'N/A'):<12}"
                  f"{info.get('omics', 'N/A'):<20}{info.get('drug', 'N/A'):<20}{info.get('components', 'N/A'):<25}")
        
        print(f"{self.your_model_name:<15}{'2025':<8}{'PyTorch':<12}{'CNV, Mu':<20}{'Molecular Graph':<20}{'GIN, Batchnorm, Dropout':<25}")
        print("="*90)
    
    def export_results(self, output_dir: str = None):
        """Xuất kết quả"""
        if output_dir is None:
            output_dir = Config.DATA_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        # Comparison data
        all_comparisons = []
        for dataset in ['CCLE', 'gCSI']:
            df = self.create_comparison_df(dataset, 'within')
            if not df.empty:
                all_comparisons.append(df)
        
        for dataset in ['CCLE-gCSI', 'gCSI-CCLE']:
            df = self.create_comparison_df(dataset, 'cross')
            if not df.empty:
                all_comparisons.append(df)
        
        if all_comparisons:
            combined = pd.concat(all_comparisons, ignore_index=True)
            combined.to_csv(f"{output_dir}/benchmark_comparison_3models.csv", index=False, float_format='%.4f')
            print(f"  ✓ benchmark_comparison_3models.csv")
        
        # Your results
        if not self.your_df.empty:
            self.your_df.to_csv(f"{output_dir}/fraGraphDRP_results.csv", index=False, float_format='%.6f')
            print(f"  ✓ fraGraphDRP_results.csv")
        
        print(f"\n💾 All data exported to: {output_dir}/")
    
    def generate_report(self, output_dir: str = None):
        """Tạo báo cáo"""
        if output_dir is None:
            output_dir = Config.REPORTS_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"{output_dir}/fraGraphDRP_report_3models_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*90 + "\n")
            f.write(" " * 30 + "FraGraphDRP BENCHMARK REPORT\n")
            f.write(" " * 20 + "Comparison with GraphDRP, tCNNs, DeepCDR\n")
            f.write("="*90 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.your_model_name}\n")
            f.write(f"Benchmark: Partin et al. 2025 - 3 Models with Similar Data\n\n")
            
            # Architecture
            f.write("="*90 + "\n")
            f.write("ARCHITECTURE COMPARISON\n")
            f.write("="*90 + "\n\n")
            
            for model in ['DeepCDR', 'GraphDRP', 'tCNNs']:
                info = self.benchmark.ARCHITECTURE_INFO.get(model, {})
                f.write(f"{model}:\n")
                f.write(f"  Year: {info.get('year', 'N/A')}\n")
                f.write(f"  Framework: {info.get('framework', 'N/A')}\n")
                f.write(f"  Omics: {info.get('omics', 'N/A')}\n")
                f.write(f"  Drug: {info.get('drug', 'N/A')}\n")
                f.write(f"  Components: {info.get('components', 'N/A')}\n\n")
            
            # Rankings
            for dataset in ['CCLE', 'gCSI']:
                f.write("\n" + "="*90 + "\n")
                f.write(f"RANKING - {dataset}\n")
                f.write("="*90 + "\n\n")
                
                df = self.create_comparison_df(dataset, 'within')
                if not df.empty:
                    df_sorted = df.sort_values('R2_Mean', ascending=False)
                    f.write(df_sorted.to_string(index=False) + "\n")
        
        print(f"📄 Report saved: {report_file}")
        return report_file


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    start_time = time.time()
    
    print("="*90)
    print(" " * 25 + "FraGraphDRP BENCHMARK ANALYSIS")
    print(" " * 20 + "vs GraphDRP, tCNNs, DeepCDR (3 Models)")
    print("="*90)
    
    # Create output directories
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.PLOTS_DIR, exist_ok=True)
    os.makedirs(Config.REPORTS_DIR, exist_ok=True)
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    
    # Step 1: Analyze FraGraphDRP results
    print("\n📊 Step 1: Analyzing FraGraphDRP results...")
    print("-"*90)
    
    analyzer = FraGraphDRPAnalyzer(Config.BASE_PATH)
    your_results_df = analyzer.analyze_all(Config.EXPERIMENTS)
    
    if your_results_df.empty:
        print("\n❌ No valid FraGraphDRP results found!")
        return None, None
    
    # Step 2: Create comparator
    print("\n📊 Step 2: Creating benchmark comparison...")
    print("-"*90)
    
    benchmark = BenchmarkData()
    comparator = FraGraphDRPComparator(your_results_df, benchmark)
    
    # Step 3: Print architecture comparison
    comparator.print_architecture_comparison()
    
    # Step 4: Print rankings
    print("\n📊 Step 3: Generating rankings...")
    print("-"*90)
    
    for dataset in ['CCLE', 'gCSI']:
        comparator.print_ranking(dataset, 'within')
    
    # Step 5: Generate plots
    print("\n📊 Step 4: Creating plots...")
    print("-"*90)
    
    comparator.plot_all_comparisons()
    
    # Step 6: Export results
    print("\n📊 Step 5: Exporting results...")
    print("-"*90)
    
    comparator.export_results()
    comparator.generate_report()
    
    # Final
    elapsed = time.time() - start_time
    print(f"\n✅ COMPLETE! Total time: {elapsed:.2f}s")
    print("="*90)
    print("📁 OUTPUT FILES:")
    print(f"   • {Config.PLOTS_DIR}/     - Comparison plots")
    print(f"   • {Config.REPORTS_DIR}/   - Text reports")
    print(f"   • {Config.DATA_DIR}/      - CSV data")
    print("="*90)
    
    return your_results_df, comparator


if __name__ == "__main__":
    your_df, comparator = main()