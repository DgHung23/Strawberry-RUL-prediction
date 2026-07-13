"""
Generate the comprehensive PDF Report: Strawberry RUL Prediction Improvement Suggestions.
"""
from fpdf import FPDF
from pathlib import Path

class RULReport(FPDF):
    def __init__(self):
        super().__init__('P', 'mm', 'A4')
        # Register fonts that support Unicode (Vietnamese chars)
        self.add_font('Arial', '', r'C:\Windows\Fonts\arial.ttf', uni=True)
        self.add_font('Arial', 'B', r'C:\Windows\Fonts\arialbd.ttf', uni=True)
        self.add_font('ArialMono', '', r'C:\Windows\Fonts\cour.ttf', uni=True)
        self.add_font('ArialMono', 'B', r'C:\Windows\Fonts\courbd.ttf', uni=True)
        self.set_auto_page_break(True, 15)

    def header(self):
        if self.page_no() == 1:
            return  # Skip header on cover page
        self.set_font('Arial', 'B', 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, 'Strawberry RUL Prediction — Improvement Report', align='L')
        self.cell(0, 5, f'Page {self.page_no()}', align='R', new_x="LMARGIN", new_y="NEXT")
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-15)
        self.set_font('Arial', '', 7)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, 'Generated 2026-07-08 | Strawberry RUL Prediction Project', align='C')

    def title_page(self):
        self.add_page()
        self.ln(25)
        # Title block
        self.set_fill_color(220, 30, 30)
        self.rect(15, 35, self.w - 30, 45, 'F')
        self.set_font('Arial', 'B', 28)
        self.set_text_color(255, 255, 255)
        self.set_y(42)
        self.cell(0, 15, 'STRAWBERRY RUL PREDICTION', align='C', new_x="LMARGIN", new_y="NEXT")
        self.set_font('Arial', 'B', 22)
        self.cell(0, 12, 'IMPROVEMENT ANALYSIS REPORT', align='C', new_x="LMARGIN", new_y="NEXT")

        self.ln(20)
        self.set_font('Arial', 'B', 14)
        self.set_text_color(50, 50, 50)
        self.cell(0, 10, 'Model Accuracy Optimization Strategies', align='C', new_x="LMARGIN", new_y="NEXT")
        self.ln(8)

        # Key metrics box
        self.set_fill_color(245, 245, 245)
        self.set_draw_color(180, 180, 180)
        box_x = 30
        box_w = self.w - 60
        self.rect(box_x, self.get_y(), box_w, 52, 'DF')

        self.set_xy(box_x + 5, self.get_y() + 4)
        self.set_font('Arial', 'B', 11)
        self.set_text_color(180, 30, 30)
        self.cell(box_w - 10, 7, 'CURRENT PERFORMANCE (Model A — Best)', align='C', new_x="LMARGIN", new_y="NEXT")

        metrics = [
            ('MAE', '43.1 hours', '~1.8 days average error'),
            ('RMSE', '49.9 hours', '~2.1 days (penalizes large errors)'),
            ('R-squared', '0.61', 'Explains 61% of variance'),
            ('EOL Error', '~90 hours', 'Predicts 90h when fruit is spoiled (RUL=0)'),
        ]
        self.set_xy(box_x + 10, self.get_y() + 3)
        for label, value, note in metrics:
            self.set_font('Arial', 'B', 10)
            self.set_text_color(50, 50, 50)
            self.cell(30, 8, f'{label}:', align='R')
            self.set_font('Arial', 'B', 12)
            self.set_text_color(200, 30, 30)
            self.cell(35, 8, value, align='C')
            self.set_font('Arial', '', 9)
            self.set_text_color(120, 120, 120)
            self.cell(0, 8, note, align='L', new_x="LMARGIN", new_y="NEXT")

        self.ln(12)
        self.set_font('Arial', '', 10)
        self.set_text_color(80, 80, 80)
        self.cell(0, 7, 'Prepared: 2026-07-08 | Project: Strawberry RUL Prediction | Models: A (EfficientNet-B0+GRU) & B (MobileNetV2+LSTM)', align='C', new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 7, 'Test Split: F05 (1 fruit, 672 sequences) | Train: F01-F04 (4 fruits, 2614 sequences)', align='C', new_x="LMARGIN", new_y="NEXT")

    def section_title(self, title):
        self.ln(5)
        self.set_fill_color(220, 30, 30)
        self.set_text_color(255, 255, 255)
        self.set_font('Arial', 'B', 14)
        h = 9
        self.cell(0, h, f'  {title}', fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(4)

    def sub_title(self, title):
        self.set_font('Arial', 'B', 11)
        self.set_text_color(50, 50, 50)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font('Arial', '', 9.5)
        self.set_text_color(60, 60, 60)
        self.multi_cell(0, 5.5, text, align='L')
        self.ln(1)

    def bullet(self, text, indent=5):
        self.set_font('Arial', '', 9.5)
        self.set_text_color(60, 60, 60)
        x0 = self.l_margin + indent
        self.set_x(x0)
        self.cell(4, 5.5, '-')
        self.multi_cell(self.w - self.r_margin - x0 - 4, 5.5, text, align='L')
        self.ln(0.5)

    def code_block(self, text):
        self.set_font('ArialMono', '', 8)
        self.set_text_color(40, 40, 40)
        self.set_fill_color(248, 248, 248)
        self.set_draw_color(200, 200, 200)
        lines = text.strip().split('\n')
        block_h = len(lines) * 4.5 + 6
        if self.get_y() + block_h > self.h - 25:
            self.add_page()
        y0 = self.get_y()
        self.rect(self.l_margin + 3, y0, self.w - self.l_margin - self.r_margin - 6, block_h, 'DF')
        self.set_xy(self.l_margin + 6, y0 + 3)
        for line in lines:
            self.cell(0, 4.5, line, new_x="LMARGIN", new_y="NEXT")
            self.set_x(self.l_margin + 6)
        self.set_y(y0 + block_h + 3)

    def image_center(self, path, w=170):
        if Path(path).exists():
            if self.get_y() + 75 > self.h - 25:
                self.add_page()
            self.image(str(path), x=(self.w - w) / 2, w=w)
            self.ln(4)
        else:
            self.body_text(f'[Image not found: {path}]')

    def warning_box(self, text):
        self.set_fill_color(255, 243, 205)
        self.set_draw_color(255, 193, 7)
        self.set_font('Arial', 'B', 9)
        self.set_text_color(150, 100, 0)
        if self.get_y() + 15 > self.h - 25:
            self.add_page()
        y0 = self.get_y()
        self.rect(self.l_margin + 2, y0, self.w - self.l_margin - self.r_margin - 4, 12, 'DF')
        self.set_xy(self.l_margin + 6, y0 + 3)
        self.multi_cell(self.w - self.l_margin - self.r_margin - 16, 5, text)
        self.set_y(y0 + 14)

    def improvement_card(self, code, title, description, impact, effort, recommendations):
        """Render a single improvement suggestion card."""
        check_gap = self.get_y() + 40 > self.h - 25
        if check_gap:
            self.add_page()

        # Impact color
        impact_colors = {'HIGH': (46, 125, 50), 'MEDIUM': (230, 130, 0), 'LOW': (150, 150, 150)}
        effort_colors = {'LOW': (46, 125, 50), 'MEDIUM': (230, 130, 0), 'HIGH': (180, 30, 30)}
        imp_c = impact_colors.get(impact, (150,150,150))
        eff_c = effort_colors.get(effort, (150,150,150))

        y0 = self.get_y()
        card_w = self.w - self.l_margin - self.r_margin

        # Card background
        self.set_fill_color(250, 250, 250)
        self.set_draw_color(200, 200, 200)
        self.rect(self.l_margin, y0, card_w, 38, 'DF')

        # Code badge
        self.set_fill_color(*imp_c)
        self.set_text_color(255, 255, 255)
        self.set_font('Arial', 'B', 9)
        self.set_xy(self.l_margin + 3, y0 + 2)
        self.cell(20, 6, f' {code} ', fill=True)
        self.set_xy(self.l_margin + 26, y0 + 2)
        self.set_text_color(30, 30, 30)
        self.set_font('Arial', 'B', 11)
        self.cell(card_w - 60, 6, title)

        # Impact/Effort badges
        self.set_font('Arial', 'B', 7)
        self.set_fill_color(*imp_c)
        self.set_text_color(255, 255, 255)
        self.set_xy(self.l_margin + card_w - 50, y0 + 2)
        self.cell(22, 6, f'IMPACT: {impact}', fill=True)
        self.set_fill_color(*eff_c)
        self.set_xy(self.l_margin + card_w - 26, y0 + 2)
        self.cell(22, 6, f'EFFORT: {effort}', fill=True)

        # Description
        self.set_xy(self.l_margin + 5, y0 + 10)
        self.set_font('Arial', '', 8.5)
        self.set_text_color(60, 60, 60)
        self.multi_cell(card_w - 10, 4.5, description)

        # Recommendations
        rec_y = self.get_y() + 1
        self.set_font('Arial', 'B', 7.5)
        self.set_text_color(*imp_c)
        self.set_xy(self.l_margin + 5, rec_y)
        self.cell(card_w - 10, 5, 'How to implement:')
        self.set_font('Arial', '', 7.5)
        self.set_text_color(80, 80, 80)
        for r in recommendations:
            rec_y += 4.5
            self.set_xy(self.l_margin + 10, rec_y)
            self.multi_cell(card_w - 15, 4.5, f'  {r}')

        final_y = max(rec_y + 6, y0 + 38)
        self.set_y(final_y)
        self.ln(2)


def build_report():
    pdf = RULReport()
    project_root = Path(__file__).resolve().parents[1]
    img_dir = project_root / 'output' / 'reports' / 'improvement_analysis'

    # ==================== COVER PAGE ====================
    pdf.title_page()

    # ==================== SECTION 1: EXECUTIVE SUMMARY ====================
    pdf.add_page()
    pdf.section_title('1. EXECUTIVE SUMMARY')

    pdf.body_text(
        'This report provides a comprehensive analysis of the current Strawberry RUL (Remaining Useful Life) '
        'prediction models and proposes multiple strategies to improve accuracy. The current best model (Model A: '
        'EfficientNet-B0 + CBAM + GRU) achieves an MAE of 43.1 hours (~1.8 days) with an R-squared of 0.61. '
        'While this is a reasonable prototype result, the error margin of ~2 days is too large for practical '
        'applications in supply chain or quality control scenarios.'
    )

    pdf.sub_title('Key Problems Identified')
    pdf.body_text(
        'Through systematic analysis of training dynamics, prediction patterns, data pipeline, labeling protocol, '
        'and model architecture, we identified 6 major problem areas:'
    )

    problems_summary = [
        '1. EOL FAILURE: The model predicts ~90 hours RUL when the fruit is already at end-of-life (actual RUL = 0). '
        'This is the most critical issue — the model fundamentally fails to recognize spoiled fruit.',
        '2. SEVERE OVERFITTING: Validation loss increases from epoch 4 onwards (from 9.5h to 19h) while training '
        'loss continues dropping (from 10.5h to 5.4h). The generalization gap exceeds 13 hours.',
        '3. SYSTEMATIC BIAS: The model under-predicts RUL for fresh fruit (~55h too low) and massively over-predicts '
        'RUL for spoiled fruit (~90h too high). The output range is compressed to ~160-190h regardless of actual RUL.',
        '4. LIMITED DATA: Only 6 strawberry fruits total, with 4 used for training (~2,614 sequences). All 6 fruits '
        'share the same EOL timestamp, contradicting the labeling protocol.',
        '5. TRAINING DEFICIENCIES: No early stopping, no learning rate scheduling, no data augmentation, batch size '
        'of only 4, and only 10 training epochs.',
        '6. ARCHITECTURE LIMITATIONS: Single-layer unidirectional GRU, only last timestep used, only 2 environmental '
        'features (firmness data available but unused), crude normalization.'
    ]
    for p in problems_summary:
        pdf.bullet(p)

    # ==================== SECTION 2: CURRENT MODEL PERFORMANCE ====================
    pdf.section_title('2. CURRENT MODEL PERFORMANCE ANALYSIS')

    pdf.sub_title('2.1 Performance Metrics')
    pdf.body_text(
        'Both Model A (EfficientNet-B0 + CBAM + GRU) and Model B (MobileNetV2 + CBAM + LSTM) were evaluated on '
        'the held-out test fruit F05 (672 sequences). Model A significantly outperforms Model B, confirming that '
        'the CNN backbone choice (EfficientNet vs MobileNet) matters more than the RNN choice (GRU vs LSTM).'
    )

    pdf.image_center(img_dir / '04_error_data_analysis.png', w=175)

    pdf.sub_title('2.2 Predictions vs Actual Values')
    pdf.body_text(
        'The scatter plot below reveals the fundamental problem: points deviate dramatically from the perfect '
        'prediction line. Fresh fruit (RUL > 200h) is under-predicted by ~55h, while spoiled fruit (RUL = 0) '
        'is predicted at ~80-110h. The model essentially collapses to predicting a narrow range (~160-190h) '
        'regardless of the actual RUL value.'
    )

    pdf.image_center(img_dir / '01_pred_vs_actual.png', w=175)

    pdf.sub_title('2.3 Training Dynamics & Overfitting')
    pdf.body_text(
        'The training curves show a classic overfitting pattern. Both models achieve their best validation loss '
        'around epoch 3-4, after which validation loss diverges upward while training loss continues to decrease. '
        'By epoch 9, Model A has a generalization gap of 13.4 hours — the model is memorizing training fruit '
        'patterns that do not transfer to unseen fruit.'
    )

    pdf.image_center(img_dir / '02_training_overfitting.png', w=175)

    # ==================== SECTION 3: EOL FAILURE DEEP DIVE ====================
    pdf.section_title('3. EOL FAILURE — DEEP DIVE ANALYSIS')

    pdf.warning_box(
        'CRITICAL ISSUE: The model cannot recognize spoiled fruit. When actual RUL = 0 (fruit at end-of-life), '
        'Model A predicts a mean of ~80 hours and Model B predicts ~100 hours. This makes the model unusable '
        'for practical spoilage detection.'
    )

    pdf.sub_title('3.1 Root Causes of EOL Failure')
    pdf.body_text(
        'Three interconnected root causes were identified for the EOL prediction failure:'
    )

    pdf.bullet(
        'RUL CLIPPING IN LABELING (label_rul.py:108-112): The labeling script clips all negative RUL values to 0 '
        'using .clip(lower=0). This means the model NEVER sees negative RUL values during training, so it has '
        'no signal to learn that "past EOL" is a distinct state from "exactly at EOL." The model cannot distinguish '
        'between a fruit that just reached EOL (RUL=0) and one that spoiled 3 days ago (which should have RUL=-72).'
    )
    pdf.bullet(
        'IDENTICAL EOL FOR ALL FRUITS (eol.py:10-15): All 6 fruits (F01-F06) share the exact same EOL timestamp '
        '"2026-03-26 08:00:00." This contradicts the labeling protocol which states "Each fruit has its own EOL '
        'timestamp." In reality, fruits spoil at different rates. Using a shared EOL means the model learns an '
        'average decay rate rather than fruit-specific patterns.'
    )
    pdf.bullet(
        'ZERO-VARIANCE TARGETS AT EOL: With clipped RUL values, all post-EOL frames have RUL=0 as their target. '
        'But visually, a fruit 1 day past EOL looks different from a fruit 3 days past EOL. The model receives '
        'conflicting signals: different visual inputs mapping to the same target value.'
    )

    pdf.image_center(img_dir / '03_eol_analysis.png', w=175)

    pdf.sub_title('3.2 Proposed Solutions for EOL Failure')
    pdf.body_text('The following approaches are ordered from simplest to most sophisticated:')

    pdf.bullet(
        'SOLUTION 1 — Allow Negative RUL: Remove the .clip(lower=0) in label_rul.py. Allow RUL to go negative '
        'for post-EOL frames. This gives the model a continuous target range and teaches it to distinguish '
        '"recently spoiled" from "long spoiled." Impact: HIGH, Effort: LOW.'
    )
    pdf.bullet(
        'SOLUTION 2 — Per-Fruit EOL Timestamps: Assign individual EOL timestamps to each fruit based on visual '
        'inspection. Different fruits spoil at different rates; the model needs to learn this variation. '
        'Impact: MEDIUM, Effort: MEDIUM.'
    )
    pdf.bullet(
        'SOLUTION 3 — Multi-Task Learning: Add a binary classification head ("is the fruit spoiled?") alongside '
        'the RUL regression head. The classification task provides a strong gradient signal for the EOL boundary. '
        'Impact: HIGH, Effort: MEDIUM.'
    )
    pdf.bullet(
        'SOLUTION 4 — Ordinal Regression / Quantile Loss: Instead of pure MAE regression, use quantile regression '
        'or treat RUL prediction as an ordinal classification problem (predict discrete RUL buckets). '
        'Impact: MEDIUM, Effort: MEDIUM.'
    )
    pdf.bullet(
        'SOLUTION 5 — Survival Analysis Approach: Reformulate as a survival analysis problem using Cox proportional '
        'hazards or DeepSurv. This naturally handles censored data and time-to-event prediction. '
        'Impact: HIGH, Effort: HIGH.'
    )

    # ==================== SECTION 4: OVERFITTING SOLUTIONS ====================
    pdf.section_title('4. OVERFITTING — MITIGATION STRATEGIES')

    pdf.body_text(
        'The overfitting problem stems primarily from the mismatch between model capacity (4.76M parameters) and '
        'dataset size (2,614 sequences from only 4 fruits). The model has enough capacity to memorize individual '
        'fruit characteristics rather than learning generalizable spoilage patterns.'
    )

    pdf.sub_title('4.1 Immediate Regularization Improvements')

    pdf.improvement_card(
        'R1', 'Data Augmentation Pipeline',
        'Apply image augmentations to artificially increase dataset diversity. Transformations should preserve '
        'spoilage-relevant features while varying lighting, orientation, and scale.',
        'HIGH', 'LOW',
        [
            'Random rotation (+-15 degrees) — strawberries can be photographed at any angle',
            'Color jitter (brightness +-10%, contrast +-10%, saturation +-5%) — simulates lighting variation',
            'Random horizontal flip (spoilage is spatially symmetric for most patterns)',
            'RandomResizedCrop (scale 0.9-1.0) — simulates slight camera distance variation',
            'Add to dataset.py __getitem__ using torchvision.transforms.RandomApply',
        ]
    )

    pdf.improvement_card(
        'R2', 'Early Stopping + ReduceLROnPlateau',
        'Stop training when validation loss stops improving, and automatically reduce learning rate when '
        'progress plateaus. This prevents the model from entering the overfitting regime.',
        'HIGH', 'LOW',
        [
            'Add EarlyStopping(patience=5, min_delta=0.5) callback in train.py',
            'Add ReduceLROnPlateau(factor=0.5, patience=3) — halves LR when val loss stalls',
            'Save best model based on val_loss (already implemented, just need early stopping logic)',
            'Expected: training stops at epoch 4-5 instead of running to epoch 10 while overfitting',
        ]
    )

    pdf.improvement_card(
        'R3', 'Stronger Dropout + Weight Decay',
        'Increase regularization strength to reduce the model\'s ability to memorize training fruit patterns.',
        'MEDIUM', 'LOW',
        [
            'Increase dropout from 0.2 to 0.4 in both GRU and regression head',
            'Add weight_decay=1e-4 to Adam optimizer in train.py',
            'Consider SpatialDropout2D for the CNN feature maps',
            'Add Dropout after CBAM attention module (before pooling)',
        ]
    )

    pdf.improvement_card(
        'R4', 'MixUp / CutMix Augmentation',
        'MixUp creates synthetic training samples by linearly interpolating two random samples (images + labels). '
        'CutMix replaces a patch of one image with a patch from another. Both are proven to reduce overfitting '
        'in small datasets.',
        'HIGH', 'MEDIUM',
        [
            'Implement MixUp: lambda ~ Beta(0.4, 0.4), mix images and RUL labels proportionally',
            'CutMix: replace random rectangular region with another sample\'s corresponding region',
            'Apply at batch level in training loop (before model forward pass)',
            'These techniques are especially effective when each fruit has a unique visual "signature"',
        ]
    )

    pdf.improvement_card(
        'R5', 'Freeze Backbone (Transfer Learning)',
        'Freeze the EfficientNet-B0 backbone weights and only train the CBAM + GRU + regression head. This '
        'dramatically reduces trainable parameters while preserving the powerful ImageNet-pretrained features.',
        'MEDIUM', 'LOW',
        [
            'Set freeze_backbone=True in model __init__ (already implemented, just change the flag)',
            'Train only CBAM (~205K) + GRU (~660K) + Regression (~8K) = ~873K params vs 4.76M',
            'After initial convergence, optionally unfreeze last few EfficientNet blocks for fine-tuning',
            'Reduces overfitting risk by 5x based on trainable parameter count',
        ]
    )

    # ==================== SECTION 5: ARCHITECTURE IMPROVEMENTS ====================
    pdf.section_title('5. ARCHITECTURE IMPROVEMENTS')

    pdf.body_text(
        'The current architecture (EfficientNet-B0 -> CBAM -> GRU(last) -> Regression) is functional but has '
        'several design limitations that constrain its ability to model the fruit spoilage process accurately.'
    )

    pdf.improvement_card(
        'A1', 'Bidirectional GRU/LSTM (BiGRU/BiLSTM)',
        'Replace the unidirectional GRU with a bidirectional variant. A BiGRU processes the sequence both forward '
        'and backward, capturing temporal context from both directions. For RUL prediction, looking at both past '
        'AND future frames (relative to each timestep) helps the model understand the full spoilage trajectory.',
        'HIGH', 'MEDIUM',
        [
            'Replace nn.GRU with nn.GRU(bidirectional=True), hidden_size=64 to keep param count similar',
            'Output dimension doubles to 128 (64*2), feed into regression head as before',
            'Expected improvement: better temporal context, especially for mid-sequence frames',
            'Code change: ~5 lines in model.py (GRU init + output handling)',
        ]
    )

    pdf.improvement_card(
        'A2', 'Attention Pooling Over All Timesteps',
        'Instead of using only the last GRU timestep output, apply learned attention weights over ALL timesteps. '
        'This allows the model to focus on the most informative frames (e.g., when visual changes accelerate).',
        'HIGH', 'MEDIUM',
        [
            'Add attention layer: score = Linear(gru_output), weights = softmax(scores), output = weighted sum',
            'This is NOT the same as CBAM (which attends over spatial dimensions)',
            'Temporal attention tells the model "which frames in the sequence are most informative"',
            'Implementation: ~15 lines of code, significant expressive power gain',
        ]
    )

    pdf.improvement_card(
        'A3', 'Multi-Layer RNN (2-3 layers)',
        'Increase GRU depth from 1 to 2-3 layers with proper dropout between layers. Deeper RNNs can model '
        'more complex temporal dynamics (acceleration of spoilage, non-linear decay patterns).',
        'MEDIUM', 'LOW',
        [
            'Set num_layers=2 or 3 in model __init__',
            'GRU/LSTM dropout between layers activates automatically when num_layers > 1',
            'Keep hidden_size moderate (64-128) to avoid parameter explosion',
            'Monitor for increased overfitting — pair with stronger regularization',
        ]
    )

    pdf.improvement_card(
        'A4', 'Include Firmness as 3rd Environmental Feature',
        'The labeling pipeline already parses firmness data (label_rul.py imports hardness.csv). Adding firmness '
        'as a 3rd environmental feature gives the model a direct physical measurement of fruit degradation.',
        'MEDIUM', 'LOW',
        [
            'Change env_dim from 2 to 3 in model.py (temp, humidity, firmness)',
            'Update dataset.py to load and normalize firmness values',
            'Normalize: firmness / max_firmness (strawberry firmness typically 0-15 N)',
            'Firmness is a strong proxy for spoilage — it changes before visual signs appear',
        ]
    )

    pdf.improvement_card(
        'A5', 'Larger Backbone / Vision Transformer',
        'Upgrade from EfficientNet-B0 (5.3M params) to EfficientNet-B2 (9.1M) or a Vision Transformer (ViT-B/16). '
        'Larger backbones capture finer visual details critical for early spoilage detection (tiny mold spots, '
        'subtle color shifts).',
        'MEDIUM', 'HIGH',
        [
            'Replace efficientnet_b0 with efficientnet_b2 (same API, different feature dim: 1408 vs 1280)',
            'ViT-B/16: patch-based attention may be better at detecting localized mold spots',
            'Risk: larger models worsen overfitting on small dataset — pair with R1-R5 mitigations',
            'Only attempt after implementing data augmentation and regularization improvements',
        ]
    )

    pdf.image_center(img_dir / '05_architecture_limits.png', w=175)

    # ==================== SECTION 6: LOSS FUNCTION IMPROVEMENTS ====================
    pdf.section_title('6. LOSS FUNCTION IMPROVEMENTS')

    pdf.body_text(
        'The current model uses pure L1Loss (MAE). While MAE is interpretable (error in hours), it has limitations '
        'for this specific problem: it treats all errors equally regardless of actual RUL value, and it provides '
        'no special handling for the critical EOL boundary.'
    )

    pdf.improvement_card(
        'L1', 'Huber Loss (Smooth L1)',
        'Huber loss combines MAE (for large errors) and MSE (for small errors). It gives stronger gradients for '
        'small errors (better convergence) while remaining robust to outliers.',
        'MEDIUM', 'LOW',
        [
            'Replace criterion = nn.L1Loss() with nn.HuberLoss(delta=10.0)',
            'Delta=10 means: MSE-like behavior for errors < 10h, MAE-like for larger errors',
            'One-line code change in train.py, immediate benefit for convergence stability',
        ]
    )

    pdf.improvement_card(
        'L2', 'Weighted Loss — Penalize EOL Errors More',
        'Apply higher loss weight to samples where actual RUL is near 0. This directly addresses the EOL failure '
        'by forcing the model to prioritize accuracy on spoiled/near-spoiled fruit.',
        'HIGH', 'LOW',
        [
            'Compute per-sample weight: w = 1.0 + alpha * exp(-actual_rul / beta)',
            'Alpha=2.0, beta=20.0 gives 3x weight for RUL=0 vs 1x for RUL=200',
            'Multiply loss by weight before backpropagation',
            'This is a form of "cost-sensitive learning" — errors at EOL cost more',
        ]
    )

    pdf.improvement_card(
        'L3', 'Multi-Task Loss: Regression + Classification',
        'Add an auxiliary binary classification task: "is the fruit at EOL?" The combined loss helps the model '
        'learn a sharper decision boundary around RUL=0.',
        'HIGH', 'MEDIUM',
        [
            'Add classification head: Linear(128->1) with BCE loss',
            'Target = 1 if actual_rul <= 0 else 0 (or use threshold like 24h)',
            'Combined loss: L_total = L_mae + lambda * L_bce',
            'Classification gradient helps pull EOL predictions toward zero',
        ]
    )

    pdf.improvement_card(
        'L4', 'Mean Squared Logarithmic Error (MSLE)',
        'MSLE penalizes under-prediction more heavily than over-prediction at small values. Since under-predicting '
        'fresh fruit is worse than over-predicting (false alarm vs missed detection), MSLE aligns incentives.',
        'LOW', 'MEDIUM',
        [
            'MSLE = mean((log(pred+1) - log(actual+1))^2)',
            'Handles the wide dynamic range (0-240h) better than MAE',
            'Alternative: use as secondary loss alongside MAE',
        ]
    )

    # ==================== SECTION 7: DATA & LABELING IMPROVEMENTS ====================
    pdf.section_title('7. DATA & LABELING IMPROVEMENTS')

    pdf.body_text(
        'The fundamental limitation of this project is data quantity. Six fruits is extremely small for deep '
        'learning. While architectural and training improvements help, the most impactful changes will come '
        'from better data and more accurate labeling.'
    )

    pdf.improvement_card(
        'D1', 'Per-Fruit EOL Timestamps',
        'Replace the shared EOL timestamp (2026-03-26 08:00:00 for all 6 fruits) with individual EOL timestamps '
        'determined by visual inspection of each fruit\'s final frames.',
        'HIGH', 'MEDIUM',
        [
            'Review video frames for each fruit individually near the end of recording',
            'Document the exact frame/timestamp where each fruit FIRST shows definitive spoilage',
            'Update eol.py to read per-fruit timestamps from a CSV instead of hardcoding one value',
            'This follows the labeling protocol: "Each fruit has its own EOL timestamp"',
            'Expected: more accurate RUL labels, better training signal',
        ]
    )

    pdf.improvement_card(
        'D2', 'Acquire More Fruit Data',
        'The single most impactful improvement: increase from 6 to 20-30 fruits. More fruits provide more '
        'variation in spoilage patterns, initial quality, and environmental response.',
        'HIGH', 'HIGH',
        [
            'Target: minimum 20 fruits for training (5x current), 5 for validation, 5 for test',
            'Include fruits with different initial conditions (ripeness levels)',
            'Vary storage conditions if possible (different temp/humidity combinations)',
            'Even 12 fruits (doubling current) would significantly improve generalization',
        ]
    )

    pdf.improvement_card(
        'D3', 'Sequence Length Tuning',
        'The current seq_len=5 (covering a few hours of spoilage) may be too short to capture meaningful '
        'temporal patterns. Longer sequences provide more context about spoilage rate.',
        'MEDIUM', 'LOW',
        [
            'Experiment with seq_len=[10, 15, 20] to find the optimal temporal window',
            'Longer sequences reduce the total number of samples but provide richer temporal context',
            'Trade-off: more temporal context vs fewer training samples',
            'Consider variable-length sequences with padding/masking',
        ]
    )

    pdf.improvement_card(
        'D4', 'Proper Feature Normalization',
        'Replace crude normalization (temp/30, humidity/100) with z-score normalization using training set '
        'statistics. This ensures all features have zero mean and unit variance.',
        'LOW', 'LOW',
        [
            'Compute mean and std of temp, humidity, and firmness from training set only',
            'Normalize: (value - mean) / std for each feature',
            'Save normalization parameters alongside the model checkpoint',
            'Apply same normalization during inference',
        ]
    )

    # ==================== SECTION 8: ENSEMBLE & ADVANCED STRATEGIES ====================
    pdf.section_title('8. ENSEMBLE & ADVANCED STRATEGIES')

    pdf.improvement_card(
        'E1', 'Model Ensemble (Average All 4 Variants)',
        'Average predictions from all 4 model variants (A, B, C, D). Ensemble predictions typically outperform '
        'any single model by canceling out individual model biases.',
        'HIGH', 'LOW',
        [
            'Train all 4 models with the improvements above',
            'At inference: pred = mean(pred_A, pred_B, pred_C, pred_D)',
            'Weighted ensemble: weight models by validation R-squared',
            'Already have the comparison infrastructure from compare_models.py',
        ]
    )

    pdf.improvement_card(
        'E2', 'Self-Supervised Pretraining',
        'Pretrain the CNN backbone on the fruit images themselves (without labels) using a self-supervised task '
        'like SimCLR, BYOL, or MAE. This helps the backbone learn fruit-specific visual features before fine-tuning '
        'on RUL prediction with limited labels.',
        'HIGH', 'HIGH',
        [
            'Collect all fruit images (train+val+test) for pretraining (no labels needed)',
            'Use SimCLR or BYOL: learn representations by contrasting augmented views of same image',
            'Fine-tune the pretrained backbone on RUL prediction task',
            'Especially valuable when labeled data is scarce but unlabeled images are abundant',
        ]
    )

    pdf.improvement_card(
        'E3', 'Multi-Task: RUL + Spoilage Stage Classification',
        'Predict RUL AND classify the fruit into discrete spoilage stages (fresh, early decay, moderate decay, '
        'severe decay, spoiled). The shared representation benefits both tasks.',
        'MEDIUM', 'MEDIUM',
        [
            'Define 5 stages: Fresh (>168h), Early (72-168h), Mid (24-72h), Late (0-24h), Spoiled (=<0h)',
            'Shared CNN+GRU encoder, two heads: regression (RUL) + classification (stage)',
            'Loss = L_mae + lambda * L_cross_entropy',
            'Classification provides interpretable output alongside continuous RUL',
        ]
    )

    pdf.improvement_card(
        'E4', 'Test-Time Augmentation (TTA)',
        'At inference, apply multiple augmented versions of each input, predict RUL for each, and average the '
        'results. TTA reduces prediction variance without any model changes.',
        'LOW', 'LOW',
        [
            'For each test sequence: create K augmented versions (flips, slight rotations)',
            'Predict RUL for each version, return the mean',
            'Typical K=5-10, no training changes needed',
            'Reduces prediction noise at the cost of K x inference time',
        ]
    )

    # ==================== SECTION 9: PRIORITIZED ROADMAP ====================
    pdf.add_page()
    pdf.section_title('9. PRIORITIZED IMPROVEMENT ROADMAP')

    pdf.body_text(
        'The following roadmap organizes all suggestions by priority and expected impact. Each phase builds '
        'on the improvements from the previous phase. Start with Phase 1 for immediate gains (~2-3 days of work), '
        'then proceed to Phase 2 and 3 as resources permit.'
    )

    # Phase 1
    pdf.sub_title('Phase 1: Quick Wins (Expected: Reduce MAE by 10-15 hours)')
    pdf.body_text('These changes require minimal code modifications and can be implemented in 1-3 days:')
    phase1_items = [
        'R1: Add data augmentation (rotation, color jitter) to dataset.py — ~20 lines',
        'R2: Implement EarlyStopping + ReduceLROnPlateau in train.py — ~15 lines',
        'R3: Increase dropout to 0.4, add weight_decay=1e-4 — ~3 lines',
        'R5: Set freeze_backbone=True for first 5 epochs, then unfreeze — ~1 line flag change',
        'L1: Switch from L1Loss to HuberLoss(delta=10.0) — ~1 line change',
        'D4: Use z-score normalization for environmental features — ~10 lines',
        'L2: Add weighted loss penalizing EOL errors — ~15 lines',
    ]
    for item in phase1_items:
        pdf.bullet(item)

    pdf.ln(3)

    # Phase 2
    pdf.sub_title('Phase 2: Structural Improvements (Expected: Reduce MAE by 10-20 more hours)')
    pdf.body_text('These changes require more code but address fundamental architecture limitations:')
    phase2_items = [
        'A1: Replace unidirectional GRU with BiGRU — ~5 lines in model.py',
        'A2: Add temporal attention pooling over all timesteps — ~15 lines in model.py',
        'A4: Include firmness as 3rd environmental feature — ~10 lines across model.py, dataset.py',
        'L3: Add binary EOL classification head (multi-task learning) — ~30 lines',
        'D1: Assign per-fruit EOL timestamps (requires relabeling data) — data work',
        'E1: Ensemble predictions from all 4 trained models — inference script only',
        'R4: Implement MixUp augmentation at batch level — ~20 lines in train.py',
    ]
    for item in phase2_items:
        pdf.bullet(item)

    pdf.ln(3)

    # Phase 3
    pdf.sub_title('Phase 3: Advanced (Expected: Reduce MAE to 15-25 hours)')
    pdf.body_text('These changes require significant effort but could dramatically improve accuracy:')
    phase3_items = [
        'D2: Acquire more fruit data (target: 20+ fruits) — most impactful single change',
        'A5: Upgrade to EfficientNet-B2 or ViT backbone (only after more data)',
        'E2: Self-supervised pretraining on fruit images (SimCLR/BYOL/MAE)',
        'E3: Multi-task learning with spoilage stage classification',
        'E5: Survival analysis formulation (DeepSurv / Cox PH)',
        'A3: Multi-layer RNN (2-3 layers) with proper inter-layer dropout',
    ]
    for item in phase3_items:
        pdf.bullet(item)

    pdf.ln(5)

    # Summary graphic
    pdf.image_center(img_dir / '06_improvement_map.png', w=175)

    # ==================== SECTION 10: IMPLEMENTATION DETAILS ====================
    pdf.add_page()
    pdf.section_title('10. IMPLEMENTATION GUIDELINES')

    pdf.sub_title('10.1 Code Changes: Data Augmentation (Phase 1 — R1)')
    pdf.body_text(
        'Add the following transforms to StrawberrySequenceDataset.__getitem__ in dataset.py. '
        'Apply augmentation ONLY during training, not validation/testing:'
    )
    pdf.code_block(
        '# In dataset.py __init__, add:\n'
        'self.train_transform = transforms.Compose([\n'
        '    transforms.RandomRotation(degrees=15),\n'
        '    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),\n'
        '    transforms.RandomHorizontalFlip(p=0.5),\n'
        '    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),\n'
        '])\n'
        '\n'
        '# In __getitem__, apply conditionally:\n'
        'if self.training:\n'
        '    img = self.train_transform(img)\n'
        'img = self.transform(img)  # existing normalization'
    )

    pdf.sub_title('10.2 Code Changes: Early Stopping (Phase 1 — R2)')
    pdf.body_text('Add early stopping logic to the training loop in train.py:')
    pdf.code_block(
        '# Before training loop:\n'
        'patience = 5\n'
        'patience_counter = 0\n'
        'best_val_loss = float("inf")\n'
        'scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3)\n'
        '\n'
        '# In epoch loop, after validation:\n'
        'scheduler.step(val_loss)\n'
        'if val_loss < best_val_loss - 0.5:  # min_delta\n'
        '    best_val_loss = val_loss\n'
        '    patience_counter = 0\n'
        '    torch.save(model.state_dict(), "best_model.pth")\n'
        'else:\n'
        '    patience_counter += 1\n'
        '    if patience_counter >= patience:\n'
        '        print(f"Early stopping at epoch {epoch+1}")\n'
        '        break'
    )

    pdf.sub_title('10.3 Code Changes: Weighted EOL Loss (Phase 1 — L2)')
    pdf.body_text('Add per-sample loss weighting to prioritize accuracy near EOL:')
    pdf.code_block(
        '# In train.py training loop, after computing loss:\n'
        '# Compute per-sample weight based on actual RUL\n'
        'actual_rul = ruls.squeeze()  # (batch_size,)\n'
        'eol_weight = 1.0 + 2.0 * torch.exp(-actual_rul / 20.0)\n'
        '  # weight=3.0 when RUL=0, weight~1.0 when RUL=200\n'
        'weighted_loss = (loss_per_sample * eol_weight).mean()\n'
        'weighted_loss.backward()'
    )

    pdf.sub_title('10.4 Code Changes: BiGRU + Temporal Attention (Phase 2 — A1, A2)')
    pdf.body_text('Replace the GRU with a bidirectional variant and add temporal attention:')
    pdf.code_block(
        '# In model.py __init__:\n'
        'self.gru = nn.GRU(\n'
        '    input_size=self.rnn_input_size,\n'
        '    hidden_size=rnn_hidden_size // 2,  # half for bidirectional\n'
        '    num_layers=num_layers,\n'
        '    batch_first=True,\n'
        '    bidirectional=True,\n'
        '    dropout=dropout if num_layers > 1 else 0,\n'
        ')\n'
        '# Temporal attention weights\n'
        'self.temporal_attn = nn.Linear(rnn_hidden_size, 1)\n'
        '\n'
        '# In forward(), after GRU:\n'
        'gru_out, _ = self.gru(fused_features)  # (B, S, hidden)\n'
        'attn_scores = self.temporal_attn(gru_out).squeeze(-1)  # (B, S)\n'
        'attn_weights = torch.softmax(attn_scores, dim=1)  # (B, S)\n'
        'context = torch.sum(gru_out * attn_weights.unsqueeze(-1), dim=1)\n'
        '# context replaces gru_out[:, -1, :]\n'
        'rul = self.regressor(context)'
    )

    pdf.sub_title('10.5 Code Changes: Multi-Task Head (Phase 2 — L3)')
    pdf.body_text('Add a binary classification head for EOL detection:')
    pdf.code_block(
        '# In model.py __init__:\n'
        'self.eol_classifier = nn.Sequential(\n'
        '    nn.Linear(rnn_hidden_size, 32),\n'
        '    nn.ReLU(),\n'
        '    nn.Linear(32, 1),  # binary: spoiled or not\n'
        ')\n'
        '\n'
        '# In forward(), return both:\n'
        'rul = self.regressor(context)\n'
        'eol_logit = self.eol_classifier(context)\n'
        'return rul, eol_logit\n'
        '\n'
        '# In train.py, compute combined loss:\n'
        'rul_pred, eol_logit = model(images, envs)\n'
        'loss_reg = criterion_reg(rul_pred, ruls)\n'
        'eol_target = (ruls <= 0).float().unsqueeze(1)\n'
        'loss_cls = criterion_cls(eol_logit, eol_target)\n'
        'loss = loss_reg + 0.1 * loss_cls'
    )

    # ==================== SECTION 11: EXPECTED IMPROVEMENTS ====================
    pdf.add_page()
    pdf.section_title('11. EXPECTED IMPROVEMENT TRAJECTORY')

    pdf.body_text(
        'The following table estimates the expected performance improvement as each phase is implemented. '
        'These are conservative estimates based on typical improvements reported in similar computer vision '
        'regression tasks with small datasets.'
    )

    # Expected improvements table
    pdf.set_font('Arial', 'B', 9)
    col_w = [50, 35, 35, 35, 35]
    headers = ['Phase / Change', 'MAE (h)', 'RMSE (h)', "R-squared", 'EOL Error (h)']
    pdf.set_fill_color(50, 50, 50)
    pdf.set_text_color(255, 255, 255)
    for i, (h, w) in enumerate(zip(headers, col_w)):
        pdf.cell(w, 7, h, fill=True, align='C')
    pdf.ln()

    rows = [
        ('Current (Model A)', '43.1', '49.9', '0.61', '~90'),
        ('Phase 1: Quick Wins', '28-33', '35-40', '0.70-0.75', '~40-50'),
        ('Phase 2: Structural', '18-25', '25-32', '0.78-0.84', '~15-25'),
        ('Phase 3: Advanced', '10-18', '15-25', '0.85-0.92', '~5-10'),
        ('TARGET (Production)', '<12', '<15', '>0.90', '<5'),
    ]

    for row_idx, row in enumerate(rows):
        if row_idx % 2 == 0:
            pdf.set_fill_color(245, 245, 245)
        else:
            pdf.set_fill_color(255, 255, 255)

        pdf.set_text_color(60, 60, 60)
        if row_idx == 0:
            pdf.set_font('Arial', 'B', 8.5)
            pdf.set_text_color(200, 30, 30)
        elif row_idx == len(rows) - 1:
            pdf.set_font('Arial', 'B', 8.5)
            pdf.set_text_color(30, 150, 30)
        else:
            pdf.set_font('Arial', '', 8.5)

        for val, w in zip(row, col_w):
            pdf.cell(w, 6.5, val, fill=True, align='C')
        pdf.ln()

    pdf.ln(6)

    pdf.sub_title('Key Assumptions')
    pdf.bullet('Phase 1 improvements come primarily from reduced overfitting (early stopping, augmentation) and better optimization (Huber loss, EOL weighting).')
    pdf.bullet('Phase 2 improvements come from the BiGRU providing bidirectional temporal context and the multi-task head providing explicit EOL signal.')
    pdf.bullet('Phase 3 improvements come from more data (most impactful), better backbone, and advanced pretraining.')
    pdf.bullet('The EOL Error is expected to improve the most dramatically because the current architecture has NO mechanism to distinguish EOL vs non-EOL beyond the continuous RUL regression — the multi-task classification head directly addresses this.')

    # ==================== SECTION 12: MONITORING & VALIDATION ====================
    pdf.section_title('12. MONITORING & VALIDATION PROTOCOL')

    pdf.body_text(
        'To ensure improvements are genuine and not due to chance, follow this validation protocol for each change:'
    )

    pdf.bullet('1. ISOLATE CHANGES: Test each improvement independently before combining. This identifies which changes actually help vs which are neutral or harmful.')
    pdf.bullet('2. FIXED SEEDS: Set random seeds (torch.manual_seed, np.random.seed) for reproducibility. Run each experiment 3 times with different seeds and report mean ± std.')
    pdf.bullet('3. TRACK METRICS: For each experiment, record MAE, RMSE, R-squared, and most importantly, per-RUL-range errors (0-24h, 24-72h, 72-120h, 120-168h, 168-260h).')
    pdf.bullet('4. VISUAL INSPECTION: Plot predicted vs actual for each experiment. A lower MAE is meaningless if the EOL predictions haven\'t improved.')
    pdf.bullet('5. ABLATION STUDIES: Once all improvements are combined, run ablation studies removing one component at a time to measure its marginal contribution.')
    pdf.bullet('6. TEST SET INTEGRITY: NEVER use F05 (test fruit) for hyperparameter tuning. Use only F06 (validation fruit) for all tuning decisions. Reserve F05 for final evaluation only.')

    pdf.ln(4)

    pdf.warning_box(
        'IMPORTANT: Do NOT tune hyperparameters using test set (F05) performance. The test set must remain '
        'completely untouched until final evaluation. All tuning decisions should use validation set (F06) only. '
        'Violating this rule produces overly optimistic results that won\'t generalize to new fruit.'
    )

    # ==================== SECTION 13: CONCLUSION ====================
    pdf.add_page()
    pdf.section_title('13. CONCLUSION')

    pdf.body_text(
        'The Strawberry RUL prediction project has a solid foundation: a well-structured data pipeline, a sound '
        'architectural design (CNN + Attention + RNN), proper fruit-ID-safe splits, and thorough documentation. '
        'The current MAE of 43.1 hours on the test fruit represents a functional prototype.'
    )

    pdf.body_text(
        'However, the model suffers from three critical issues that make it unsuitable for practical deployment:'
    )

    pdf.bullet('It cannot recognize spoiled fruit (predicts ~90h RUL when the fruit is already at EOL).')
    pdf.bullet('It severely overfits the 4 training fruits (generalization gap >13 hours).')
    pdf.bullet('It compresses its output range, failing to distinguish fresh fruit from aging fruit.')

    pdf.body_text(
        'The 21 improvement suggestions in this report are organized into three phases, prioritized by impact-to-effort '
        'ratio. Phase 1 (Quick Wins) can be implemented in 2-3 days and is expected to reduce MAE by 25-35%. '
        'Phase 2 addresses fundamental architecture limitations and is expected to cut MAE roughly in half. '
        'Phase 3 requires more data and effort but could bring the model to production-ready accuracy.'
    )

    pdf.ln(3)

    pdf.sub_title('Top 5 Highest-Priority Actions')
    top5 = [
        '1. Add data augmentation + early stopping (Phase 1, R1+R2) — prevents overfitting immediately',
        '2. Implement weighted EOL loss (Phase 1, L2) — directly addresses the EOL failure problem',
        '3. Remove RUL clipping to allow negative values (Phase 1, EOL Solution 1) — teaches model about post-EOL state',
        '4. Add BiGRU + temporal attention (Phase 2, A1+A2) — better temporal modeling',
        '5. Add multi-task EOL classification head (Phase 2, L3) — explicit EOL detection signal',
    ]
    for item in top5:
        pdf.bullet(item)

    pdf.ln(5)
    pdf.body_text(
        'The single most impactful change — acquiring more fruit data — is listed in Phase 3 due to the effort required, '
        'but if resources permit, it should be prioritized above all other changes. Deep learning models are fundamentally '
        'data-hungry; no amount of architectural cleverness can fully compensate for having only 4 training fruits.'
    )

    pdf.ln(8)
    pdf.set_font('Arial', '', 9)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(0, 6, '— End of Report —', align='C')

    # ==================== SAVE ====================
    output_path = project_root / 'output' / 'reports' / 'Strawberry_RUL_Improvement_Report.pdf'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))
    print(f'Report saved to: {output_path}')
    return output_path


if __name__ == '__main__':
    build_report()
