import csv
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import pearsonr, spearmanr, kendalltau, chi2, f
import statsmodels.api as sm
from statsmodels.stats.diagnostic import (
    het_breuschpagan,
    het_white,
    het_goldfeldquandt
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# =========================================================
# CONFIGURATION
# =========================================================

# Dataset groups
LINEARITY_DATASETS = [
    'linear_positive',
    'linear_negative',
    'quadratic',
    'scatter',
    'sigmoid',
    'exponential',
]

HETERO_DATASETS = [
    'linear_positive',
    'linear_negative',
    'hetero_positive',
    'hetero_negative',
    'bow_positive',
    'bow_negative',
]

OUTLIER_DATASETS = [
    'outlier',
]

NLDD_DATASETS = [
    'linear_positive',
]

# Put true outlier indices here only for datasets used in outlier experiments
TRUE_OUTLIERS_MAP = {
    'outlier': [10, 11, 22, 23, 27, 29, 33, 41, 45, 47]
}

DATASET_DIR = 'dataset'
OUTPUT_DIR = 'result_graph'

os.makedirs(os.path.join(OUTPUT_DIR, 'linearity'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'hetero'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'outlier'), exist_ok=True)


# =========================================================
# NLDD CLASS
# =========================================================

class NLDD:
    def __init__(
        self,
        filename,
        mode,
        true_outlier_indices=None,
        z_threshold=2.0,
        iqr_factor=1.5,
        lof_neighbour=5,
        lof_contamination=0.1,
        nldd_outlier_k=0.05
    ):
        """
        mode:
            - 'linearity' -> compare PCC, Spearman, Kendall tau, WLS, NLDD
            - 'hetero'    -> compare BPT, WT, GQT, NLDD
            - 'outlier'   -> compare Z-score, IQR, LOF, NLDD against ground truth
        """
        self.filename = filename
        self.mode = mode
        self.csv_file = os.path.join(DATASET_DIR, f'{filename}.csv')

        # Outlier config
        self.true_outliers = set(true_outlier_indices or [])
        self.z_threshold = z_threshold
        self.iqr_factor = iqr_factor
        self.lof_neighbour = lof_neighbour
        self.lof_contamination = lof_contamination
        self.nldd_outlier_k = nldd_outlier_k

        # raw / processed data
        self.dataset = []
        self.float_dataset = None
        self.sorted_dataset = None
        self.x = None
        self.y = None

        # line fit
        self.slope = None
        self.intercept = None
        self.y_hat = None
        self.abs_residuals = None

        # metric outputs
        self.pcc_result = None
        self.sr_result = None
        self.kt_result = None
        self.wls_result = None

        self.bpt_result = None
        self.wt_result = None
        self.gqt_result = None

        self.nldd_result = None
        self.abs_residual_mean = None
        self.abs_residual_std = None
        self.y_range = None

        # outlier outputs
        self.nldd_outliers = set()
        self.z_outliers = set()
        self.iqr_outliers = set()
        self.lof_outliers = set()

    # =====================================================
    # MAIN
    # =====================================================

    def main(self):
        self.load_and_prepare_data()

        if self.mode == 'linearity':
            self.compute_linearity_metrics()
            self.compute_nldd(self.x, self.y)
            self.plot_graph_linearity(
                x=self.x,
                y=self.y,
                pcc=self.pcc_result,
                sr=self.sr_result,
                kt=self.kt_result,
                wls=self.wls_result,
                nldd=self.nldd_result,
                name=self.filename
            )

        elif self.mode == 'hetero':
            self.compute_hetero_metrics()
            self.compute_nldd(self.x, self.y)
            self.plot_graph_hetero(
                x=self.x,
                y=self.y,
                bpt=self.bpt_result,
                wt=self.wt_result,
                gqt=self.gqt_result,
                nldd=self.nldd_result,
                name=self.filename
            )

        elif self.mode == 'outlier':
            self.compute_outlier_methods()

            self.plot_graph_outlier("Z-Score", self.z_outliers, f"{self.filename}_ZScore_vs_GT")
            self.plot_graph_outlier("IQR", self.iqr_outliers, f"{self.filename}_IQR_vs_GT")
            self.plot_graph_outlier("LOF", self.lof_outliers, f"{self.filename}_LOF_vs_GT")
            self.plot_graph_outlier("NLDD", self.nldd_outliers, f"{self.filename}_NLDD_vs_GT")

            self.print_eval("Z-Score", self.z_outliers)
            self.print_eval("IQR", self.iqr_outliers)
            self.print_eval("LOF", self.lof_outliers)
            self.print_eval("NLDD", self.nldd_outliers)

        elif self.mode == 'nldd_alone':
            self.compute_nldd(self.x, self.y)
            print(f"NLDD for {self.filename}: {self.nldd_result:.4f}")

        else:
            raise ValueError("mode must be either 'linearity', 'hetero', 'outlier', or 'nldd_alone'")

    # =====================================================
    # DATA LOADING
    # =====================================================

    def load_and_prepare_data(self):
        self.dataset = self.import_csv_file(self.csv_file)
        self.float_dataset = self.get_float_two_d_list(self.dataset)
        self.sorted_dataset = self.sort_two_d_array_by_column(self.float_dataset, 0)
        self.x, self.y = self.separate_two_d_array_by_column(self.sorted_dataset)

    @staticmethod
    def import_csv_file(filename):
        rows = []
        with open(filename, 'r', newline='') as csvfile:
            csvreader = csv.reader(csvfile)
            rows = list(csvreader)
        return rows[1:]  # remove header

    @staticmethod
    def get_float_two_d_list(string_two_d_list):
        return np.array(string_two_d_list, dtype=np.float64)

    @staticmethod
    def sort_two_d_array_by_column(two_d_array, column_index):
        sort_idx = np.argsort(two_d_array[:, column_index])
        return two_d_array[sort_idx]

    @staticmethod
    def separate_two_d_array_by_column(two_d_array):
        return two_d_array[:, 0], two_d_array[:, 1]

    # =====================================================
    # BASIC CORRELATIONS
    # =====================================================

    @staticmethod
    def pearson_correlation(x, y):
        corr, _ = pearsonr(x, y)
        return float(f"{corr:.4f}")

    @staticmethod
    def spearman_rank(x, y):
        corr, _ = spearmanr(x, y)
        return float(f"{corr:.4f}")

    @staticmethod
    def kendall_tau(x, y):
        corr, _ = kendalltau(x, y)
        return float(f"{corr:.4f}")

    # =====================================================
    # WLS-BASED CORRELATION
    # =====================================================

    @staticmethod
    def weighted_least_squares_correlation(x, y, eps=1e-8):
        """
        Residual-based weighted correlation.
        Numerically stabilized to avoid exploding weights.
        """
        X = sm.add_constant(x)
        ols = sm.OLS(y, X).fit()
        residuals = ols.resid

        # Stabilized inverse-residual-square weights
        resid_sq = np.maximum(residuals ** 2, eps)

        # Optional clipping to reduce domination by a few tiny residuals
        cap = np.quantile(resid_sq, 0.95)
        resid_sq = np.minimum(resid_sq, cap)

        weights = 1.0 / resid_sq
        weights = weights / np.sum(weights)

        x_mean = np.sum(weights * x)
        y_mean = np.sum(weights * y)

        cov_xy = np.sum(weights * (x - x_mean) * (y - y_mean))
        var_x = np.sum(weights * (x - x_mean) ** 2)
        var_y = np.sum(weights * (y - y_mean) ** 2)

        denom = np.sqrt(max(var_x * var_y, eps))
        corr = cov_xy / denom
        return float(f"{corr:.4f}")

    # =====================================================
    # HETEROSKEDASTICITY TESTS
    # =====================================================

    @staticmethod
    def fit_ols_model(x, y):
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        return model

    @staticmethod
    def breusch_pagan_test(model, alpha=0.05):
        lm_stat, _, _, _ = het_breuschpagan(model.resid, model.model.exog)

        k = model.model.exog.shape[1]
        df_bp = k - 1
        crit_val = chi2.ppf(1 - alpha, df_bp)

        stat_part = fr"$X^2$={lm_stat:.4f}"
        crit_part = fr"$\chi^2_{{crit}}$(0.95,{df_bp})={crit_val:.4f}"

        if lm_stat > crit_val:
            return f"{stat_part:<13} >  {crit_part:<35}"
        else:
            return f"{stat_part:<13} <  {crit_part:<35}"

    @staticmethod
    def white_test(model, alpha=0.05):
        lm_stat, _, _, _ = het_white(model.resid, model.model.exog)

        k = model.model.exog.shape[1] - 1
        df_white = 2 * k + (k * (k - 1)) // 2
        crit_val = chi2.ppf(1 - alpha, df_white)

        stat_part = fr"$X^2$={lm_stat:.4f}"
        crit_part = fr"$\chi^2_{{crit}}$(0.95,{df_white})={crit_val:.4f}"

        if lm_stat > crit_val:
            return f"{stat_part:<13} >  {crit_part:<35}"
        else:
            return f"{stat_part:<13} <  {crit_part:<35}"

    @staticmethod
    def goldfeld_quandt_test(model, alpha=0.05, frac=0.2, alternative="increasing"):
        y = model.model.endog
        X = model.model.exog
        n, k = X.shape

        f_stat, _, _ = het_goldfeldquandt(y, X, alternative=alternative, drop=frac)

        drop_n = int(np.floor(frac * n))
        keep = n - drop_n
        half = keep // 2
        n1 = half
        n2 = half

        df1 = n2 - k
        df2 = n1 - k

        if df1 <= 0 or df2 <= 0:
            return "GQ: insufficient df"

        Fcrit = f.ppf(1 - alpha, df1, df2)
        stat_part = fr"$F$={f_stat:.4f}"
        crit_part = fr"$F_{{crit}}$(0.95;{df1},{df2})={Fcrit:.4f}"

        if f_stat > Fcrit:
            return f"{stat_part:<12} >  {crit_part:<35}"
        else:
            return f"{stat_part:<12} <  {crit_part:<35}"

    # =====================================================
    # OUTLIER DETECTION
    # =====================================================

    @staticmethod
    def detect_outliers_z_score(y, z_threshold=2.0):
        y_mean = np.mean(y)
        y_std = np.std(y, ddof=1)

        if y_std == 0:
            return set()

        z_scores = np.abs((y - y_mean) / y_std)
        return {i for i, z in enumerate(z_scores) if z > z_threshold}

    @staticmethod
    def detect_outliers_iqr(y, iqr_factor=1.5):
        q1, q3 = np.percentile(y, [25, 75])
        iqr = q3 - q1
        lower = q1 - iqr_factor * iqr
        upper = q3 + iqr_factor * iqr
        return {i for i, val in enumerate(y) if val < lower or val > upper}

    @staticmethod
    def detect_outliers_lof(x, y, n_neighbors=5, contamination=0.1):
        X = np.column_stack([x, y])
        lof = make_pipeline(
            StandardScaler(),
            LocalOutlierFactor(
                n_neighbors=n_neighbors,
                contamination=contamination,
                novelty=False,
                n_jobs=-1
            )
        )
        labels = lof.fit_predict(X)
        return {i for i, lab in enumerate(labels) if lab == -1}

    def detect_outliers_nldd_iterative(self, threshold):
        """
        Iterative NLDD outlier extraction merged from the uploaded outlier NLDD file.
        Outliers are tracked by original row index.
        """
        x_current = np.asarray(self.x, dtype=np.float64).copy()
        y_current = np.asarray(self.y, dtype=np.float64).copy()
        current_indices = np.arange(len(x_current))
        detected = []

        iteration = 0

        while True:
            if len(x_current) < 3:
                break

            if np.unique(x_current).size <= 1:
                break

            self.compute_nldd(x_current, y_current)

            if self.nldd_result <= threshold:
                break

            if self.abs_residual_std == 0:
                break

            y_lambda = math.floor(np.max(self.abs_residuals) / self.abs_residual_std)
            if y_lambda <= 0:
                break

            buffer_value = y_lambda * self.abs_residual_std
            local_mask = self.abs_residuals > buffer_value
            local_outlier_positions = np.where(local_mask)[0]

            if len(local_outlier_positions) == 0:
                break

            original_outlier_indices = current_indices[local_outlier_positions]
            detected.extend(original_outlier_indices.tolist())

            keep_mask = ~local_mask
            x_current = x_current[keep_mask]
            y_current = y_current[keep_mask]
            current_indices = current_indices[keep_mask]

            iteration += 1

        return set(sorted(detected))

    # =====================================================
    # NLDD
    # =====================================================

    def compute_nldd(self, x_list, y_list, eps=1e-12):
        """
        NLDD = sample std of |y - y_hat| divided by range(y)
        """
        self.slope, self.intercept = self.least_square_method(x_list, y_list)
        self.y_hat = self.slope * x_list + self.intercept

        self.abs_residuals = np.abs(y_list - self.y_hat)
        self.abs_residual_mean = np.mean(self.abs_residuals)

        if len(self.abs_residuals) < 2:
            raise ValueError("At least 2 samples are needed for sample standard deviation.")

        self.abs_residual_std = np.std(self.abs_residuals, ddof=1)
        self.y_range = np.max(y_list) - np.min(y_list)

        if self.y_range <= eps:
            raise ValueError("Range of y is zero; NLDD is undefined.")

        self.nldd_result = float(f"{(self.abs_residual_std / self.y_range):.4f}")

    @staticmethod
    def least_square_method(x, y, eps=1e-12):
        x_mean = np.mean(x)
        y_mean = np.mean(y)

        x_centered = x - x_mean
        y_centered = y - y_mean

        denominator = np.sum(x_centered ** 2)
        if denominator <= eps:
            raise ValueError("Variance of x is too small for regression.")

        slope = np.sum(x_centered * y_centered) / denominator
        intercept = y_mean - slope * x_mean
        return slope, intercept

    # =====================================================
    # TASK-SPECIFIC METRICS
    # =====================================================

    def compute_linearity_metrics(self):
        self.pcc_result = self.pearson_correlation(self.x, self.y)
        self.sr_result = self.spearman_rank(self.x, self.y)
        self.kt_result = self.kendall_tau(self.x, self.y)
        self.wls_result = self.weighted_least_squares_correlation(self.x, self.y)

    def compute_hetero_metrics(self):
        model = self.fit_ols_model(self.x, self.y)
        self.bpt_result = self.breusch_pagan_test(model)
        self.wt_result = self.white_test(model)
        self.gqt_result = self.goldfeld_quandt_test(model)

    def compute_outlier_methods(self):
        self.z_outliers = self.detect_outliers_z_score(self.y, self.z_threshold)
        self.iqr_outliers = self.detect_outliers_iqr(self.y, self.iqr_factor)
        self.lof_outliers = self.detect_outliers_lof(
            self.x, self.y,
            n_neighbors=self.lof_neighbour,
            contamination=self.lof_contamination
        )
        self.nldd_outliers = self.detect_outliers_nldd_iterative(self.nldd_outlier_k)

    # =====================================================
    # VISUALIZATION
    # =====================================================

    @staticmethod
    def apply_plot_style():
        rcParams['font.family'] = 'Times New Roman'
        rcParams['font.size'] = 10
        rcParams['axes.labelsize'] = 10
        rcParams['xtick.labelsize'] = 9
        rcParams['ytick.labelsize'] = 9
        rcParams['legend.fontsize'] = 9

        rcParams['axes.linewidth'] = 0.8
        rcParams['lines.linewidth'] = 1.2

    @classmethod
    def plot_graph_linearity(cls, x, y, pcc, sr, kt, wls, nldd, name):
        """
        Preserve older style:
        - scatter only
        - large boxed results below the axis
        """
        cls.apply_plot_style()

        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        ax.scatter(x, y, s=10, color='darkblue')

        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)

        results_text = (
            f"PCC  : {pcc:.4f}\n"
            f"SRCC : {sr:.4f}\n"
            f"KTCC : {kt:.4f}\n"
            f"WLS  : {wls:.4f}\n"
            f"NLDD : {nldd:.4f}"
        )

        ax.text(
            0.5, -0.35,
            results_text,
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=9,
            fontfamily='Times New Roman',
            bbox=dict(
                facecolor='white',
                edgecolor='black',
                linewidth=0.8,
                boxstyle='square,pad=0.3'
            )
        )
        plt.subplots_adjust(bottom=0.35)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'linearity', f"{name}_linearity.png"), dpi=600, bbox_inches="tight")
        plt.show()

    @classmethod
    def plot_graph_hetero(cls, x, y, bpt, wt, gqt, nldd, name):
        """
        Preserve older style:
        - scatter only
        - boxed results below the axis
        """
        cls.apply_plot_style()

        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        ax.scatter(x, y, s=10, color='darkblue')

        ax.set_xlabel('x')
        ax.set_ylabel('y')

        ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)

        results_text = (
            f"BPT  : {bpt}\n"
            f"WT   : {wt}\n"
            f"GQT  : {gqt}\n"
            f"NLDD : {nldd:.4f}"
        )

        ax.text(
            0.5, -0.4,
            results_text,
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=9,
            bbox=dict(
                facecolor='white',
                edgecolor='black',
                linewidth=0.8,
                boxstyle='square,pad=0.3'
            )
        )
        plt.subplots_adjust(bottom=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'hetero', f"{name}_hetero.png"), dpi=600, bbox_inches="tight")
        plt.show()

    # =====================================================
    # VISUALIZATION: OUTLIER
    # =====================================================

    def plot_graph_outlier(self, method, detected_outliers, filename_suffix):
        colors = {
            'tp': '#009624',
            'fp': '#E67E22',
            'fn': '#E74C3C',
            'tn': 'blue'
        }

        fig, ax = plt.subplots(figsize=(4.8, 3.6))

        for idx, (x_val, y_val) in enumerate(zip(self.x, self.y)):
            if idx in detected_outliers and idx in self.true_outliers:
                ax.scatter(
                    x_val, y_val, color=colors['tp'], marker='*', s=90,
                    label='True Positive' if 'True Positive' not in ax.get_legend_handles_labels()[1] else ""
                )
            elif idx not in detected_outliers and idx not in self.true_outliers:
                ax.scatter(
                    x_val, y_val, color=colors['tn'], marker='o', s=20,
                    label='True Negative' if 'True Negative' not in ax.get_legend_handles_labels()[1] else ""
                )
            elif idx in detected_outliers and idx not in self.true_outliers:
                ax.scatter(
                    x_val, y_val, color=colors['fp'], marker='x', s=50,
                    label='False Positive' if 'False Positive' not in ax.get_legend_handles_labels()[1] else ""
                )
            elif idx not in detected_outliers and idx in self.true_outliers:
                ax.scatter(
                    x_val, y_val, color=colors['fn'], marker='^', s=50,
                    label='False Negative' if 'False Negative' not in ax.get_legend_handles_labels()[1] else ""
                )

        y_fit = self.slope * self.x + self.intercept
        ax.plot(self.x, y_fit, linestyle='--', color='black', linewidth=1.2, label='Best Fit Line')

        ax.set_xlabel("x", fontsize=10, fontname="Times New Roman")
        ax.set_ylabel("y", fontsize=10, fontname="Times New Roman")
        ax.tick_params(axis='both', which='major', labelsize=10)

        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontname('Times New Roman')

        ax.legend(loc='best', fontsize=10, frameon=True)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        plt.tight_layout()
        save_path = os.path.join(OUTPUT_DIR, 'outlier', f"{filename_suffix}.png")
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.show()
        plt.close(fig)

    # =====================================================
    # EVALUATION
    # =====================================================

    def print_eval(self, name, predicted):
        true_outliers = set(self.true_outliers)
        tp = predicted & true_outliers
        fp = predicted - true_outliers
        fn = true_outliers - predicted

        precision = len(tp) / (len(tp) + len(fp)) if len(tp) + len(fp) > 0 else 0.0
        recall = len(tp) / (len(tp) + len(fn)) if len(tp) + len(fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        print(f"\n{name} Evaluation:")
        print(f"  True Positives : {sorted(tp)}")
        print(f"  False Positives: {sorted(fp)}")
        print(f"  False Negatives: {sorted(fn)}")
        print(f"  Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")


# =========================================================
# RUNNER
# =========================================================

def run_linearity_group():
    print("\n========== LINEARITY DATASETS ==========")
    for filename in LINEARITY_DATASETS:
        print(f"Running linearity analysis for: {filename}")
        obj = NLDD(filename=filename, mode='linearity')
        obj.main()
        print("-" * 50)


def run_hetero_group():
    print("\n========== HETEROSKEDASTICITY DATASETS ==========")
    for filename in HETERO_DATASETS:
        print(f"Running heteroskedasticity analysis for: {filename}")
        obj = NLDD(filename=filename, mode='hetero')
        obj.main()
        print("-" * 50)


def run_outlier_group():
    print("\n========== OUTLIER DATASETS ==========")
    for filename in OUTLIER_DATASETS:
        print(f"Running outlier analysis for: {filename}")
        obj = NLDD(
            filename=filename,
            mode='outlier',
            true_outlier_indices=TRUE_OUTLIERS_MAP.get(filename, []),
            z_threshold=2.0,
            iqr_factor=1.5,
            lof_neighbour=5,
            lof_contamination=0.1,
            nldd_outlier_k=0.05
        )
        obj.main()
        print("-" * 50)

def run_nldd_alone():
    print("\n========== NLDD DATASETS ==========")
    for filename in NLDD_DATASETS:
        print(f"Running NLDD analysis for: {filename}")
        obj = NLDD(filename=filename, mode='nldd_alone')  # mode can be anything since we only call compute_nldd()
        obj.main()
        print("-" * 50)


if __name__ == '__main__':
    # run_linearity_group()
    # run_hetero_group()
    # run_outlier_group()
    run_nldd_alone()
