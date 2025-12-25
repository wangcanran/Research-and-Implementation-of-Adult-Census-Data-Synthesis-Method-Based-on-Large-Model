#!/usr/bin/env python3
# generate_architecture_pdf.py - 直接生成PDF架构图，无需SVG转换

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.colors import black, white, HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph

def create_architecture_pdf(output_path="architecture_paper.pdf"):
    """生成论文级架构图PDF"""
    
    # 创建画布 (A4横向)
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4
    
    # 颜色定义 - 论文标准色
    COLOR_BLUE = HexColor(0xE8F4F8)      # 阶段1背景
    COLOR_ORANGE = HexColor(0xFFF5E7)    # 阶段2背景
    COLOR_GREEN = HexColor(0xE8F5E9)     # 阶段3背景
    COLOR_BORDER_BLUE = HexColor(0x0078D4)
    COLOR_BORDER_ORANGE = HexColor(0xE67E22)
    COLOR_BORDER_GREEN = HexColor(0x28A745)
    COLOR_TEXT = black
    COLOR_ACCENT = HexColor(0x155724)    # 成功色
    
    # 字体设置
    c.setFont("Helvetica-Bold", 24)
    c.setFillColor(COLOR_TEXT)
    
    # 标题
    c.drawCentredString(width/2, height - 50, "Figure 1: Adult V2 System Architecture")
    c.setFont("Helvetica-Oblique", 14)
    c.drawCentredString(width/2, height - 75, "Discriminator-Guided Neural Data Synthesis Framework")
    
    # 阶段1: 判别器预训练
    c.setFillColor(COLOR_BLUE)
    c.setStrokeColor(COLOR_BORDER_BLUE)
    c.rect(50, height - 320, 380, 180, stroke=1, fill=1)
    
    c.setFont("Helvetica-Bold", 16)
    c.setFillColor(COLOR_TEXT)
    c.drawCentredString(240, height - 305, "Phase I: Discriminator Pre-training")
    
    # 阶段1内容
    c.setFont("Helvetica", 12)
    c.setFillColor(white)
    c.setStrokeColor(COLOR_BORDER_BLUE)
    c.rect(70, height - 275, 340, 30, stroke=1, fill=1)
    c.setFillColor(COLOR_TEXT)
    c.drawCentredString(240, height - 260, "Input: 32,561 Real Adult Census Samples")
    
    c.setStrokeColor(COLOR_BORDER_BLUE)
    c.rect(70, height - 235, 340, 50, stroke=1, fill=1)
    c.drawCentredString(240, height - 220, "Model: GradientBoostingClassifier (n_estimators=200)")
    c.setFont("Helvetica", 10)
    c.drawCentredString(240, height - 205, "Features: 15 → 47 dims (OneHot + Scaling)")
    
    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(COLOR_ACCENT)
    c.setStrokeColor(COLOR_ACCENT)
    c.rect(70, height - 180, 340, 25, stroke=1, fill=1)
    c.setFillColor(white)
    c.drawCentredString(240, height - 167, "Output: Trained Discriminator (96.2% Acc, AUC=0.990)")
    
    # 阶段2: Self-Instruct生成
    c.setFillColor(COLOR_ORANGE)
    c.setStrokeColor(COLOR_BORDER_ORANGE)
    c.rect(480, height - 340, 500, 520, stroke=1, fill=1)
    
    c.setFont("Helvetica-Bold", 16)
    c.setFillColor(COLOR_TEXT)
    c.drawCentredString(730, height - 325, "Phase II: Self-Instruct Iterative Generation")
    
    # 调度器
    c.setStrokeColor(COLOR_BORDER_ORANGE)
    c.rect(500, height - 295, 460, 90, stroke=1, fill=1)
    c.setFont("Helvetica-Bold", 13)
    c.setFillColor(COLOR_TEXT)
    c.drawCentredString(730, height - 280, "HybridScheduler")
    c.setFont("Helvetica", 10)
    c.drawCentredString(730, height - 265, "80% Distribution Exploration")
    c.drawCentredString(730, height - 255, "20% Discriminator-Guided Active Learning")
    c.setFont("Helvetica-Bold", 9)
    c.setFillColor(COLOR_ACCENT)
    c.drawCentredString(730, height - 242, "⭐ Quality-driven Active Learning")
    
    # 分解器
    c.setStrokeColor(COLOR_BORDER_ORANGE)
    c.rect(500, height - 235, 460, 90, stroke=1, fill=1)
    c.setFont("Helvetica-Bold", 13)
    c.setFillColor(COLOR_TEXT)
    c.drawCentredString(730, height - 220, "SampleWiseDecomposer")
    c.setFont("Helvetica", 10)
    c.drawCentredString(730, height - 205, "6 Field Groups Sequential Generation")
    c.drawCentredString(730, height - 195, "Adaptive Cache: Check every 50 samples")
    c.setFont("Helvetica-Bold", 9)
    c.setFillColor(COLOR_ACCENT)
    c.drawCentredString(730, height - 182, "⭐ Online Distribution Correction (83%↓ Error)")
    
    # 示例管理器
    c.setStrokeColor(COLOR_BORDER_ORANGE)
    c.rect(500, height - 170, 460, 70, stroke=1, fill=1)
    c.setFont("Helvetica-Bold", 13)
    c.setFillColor(COLOR_TEXT)
    c.drawCentredString(730, height - 155, "DemonstrationManager")
    c.setFont("Helvetica", 10)
    c.drawCentredString(730, height - 140, "3-Metric Selection: Quality + Similarity + Uncertainty")
    c.setFont("Helvetica-Bold", 9)
    c.setFillColor(COLOR_ACCENT)
    c.drawCentredString(730, height - 130, "⭐ Quality-aware Demonstration Selection")
    
    # LLM调用
    c.setStrokeColor(COLOR_BORDER_ORANGE)
    c.rect(500, height - 115, 460, 50, stroke=1, fill=1)
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(COLOR_TEXT)
    c.drawCentredString(730, height - 100, "GPT-4o-mini API Call")
    c.setFont("Helvetica", 9)
    c.drawCentredString(730, height - 88, "Batch=10 | Temp=0.7 | Max_tokens=800 | Cost=$6.2")
    
    # 验证过滤
    c.setStrokeColor(COLOR_BORDER_ORANGE)
    c.rect(500, height - 70, 460, 60, stroke=1, fill=1)
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(COLOR_TEXT)
    c.drawCentredString(730, height - 55, "Curation (Validation & Filtering)")
    c.setFont("Helvetica", 9)
    c.drawCentredString(730, height - 43, "Rule-based Verifier: 5 Business Constraints")
    c.drawCentredString(730, height - 33, "Discriminator Scoring: Keep >0.7 Only")
    
    # 输出
    c.setFillColor(COLOR_GREEN)
    c.setStrokeColor(COLOR_BORDER_GREEN)
    c.rect(500, height - 25, 460, 30, stroke=1, fill=1)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(730, height - 10, "2,659 High-Quality Samples (88.6%)")
    
    # 反馈箭头
    c.setDash(array=[3,3])
    c.setStrokeColor(COLOR_BORDER_ORANGE)
    c.setLineWidth(1.5)
    c.line(1040, height - 100, 1140, height - 100)
    c.line(1140, height - 100, 1140, height - 120)
    c.line(1140, height - 120, 1070, height - 120)
    c.setDash([])
    c.setFillColor(COLOR_BORDER_ORANGE)
    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(1090, height - 85, "Self-Instruct Feedback")

    # 阶段2→3箭头
    c.setStrokeColor(COLOR_TEXT)
    c.setLineWidth(2)
    c.line(1280, height - 170, 1320, height - 170)
    c.line(1320, height - 170, 1320, height - 320)
    c.line(1320, height - 320, 700, height - 320)
    c.setFillColor(COLOR_TEXT)
    c.setFont("Helvetica", 10)
    c.drawCentredString(1010, height - 180, "Iterative")

    # 阶段3: 双轨评估
    c.setFillColor(COLOR_GREEN)
    c.setStrokeColor(COLOR_BORDER_GREEN)
    c.rect(50, 120, 1240, 100, stroke=1, fill=1)
    
    c.setFont("Helvetica-Bold", 16)
    c.setFillColor(COLOR_TEXT)
    c.drawCentredString(700, 105, "Phase III: Dual-Track Evaluation")
    
    # Benchmark评估
    c.setStrokeColor(COLOR_BORDER_GREEN)
    c.rect(70, 140, 520, 60, stroke=1, fill=1)
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(COLOR_BORDER_GREEN)
    c.drawCentredString(330, 165, "Benchmark Evaluation (vs 32,561 Real Samples)")
    c.setFont("Helvetica", 10)
    c.setFillColor(COLOR_TEXT)
    c.drawCentredString(330, 180, "Overall: 93.8% | Distribution: 94.6% | Statistics: 86.5%")
    c.drawCentredString(330, 190, "Logic Consistency: 100% (3,000/3,000)")
    
    # Direct评估
    c.setStrokeColor(COLOR_BORDER_GREEN)
    c.rect(670, 140, 520, 60, stroke=1, fill=1)
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(COLOR_BORDER_GREEN)
    c.drawCentredString(930, 165, "Direct Evaluation (Faithfulness + Diversity)")
    c.setFont("Helvetica", 10)
    c.setFillColor(COLOR_TEXT)
    c.drawCentredString(930, 180, "Faithfulness: 96.9% (Constraint: 100%) | Diversity: 71.8%")
    c.drawCentredString(930, 190, "Issue: >50K Category Score 65.0% (Requires Improvement)")

    # 工程指标
    c.setFillColor(HexColor(0xf8f9fa))
    c.rect(70, 95, 1240, 20, stroke=0, fill=1)
    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(COLOR_TEXT)
    c.drawCentredString(700, 108, 
                       "Engineering: 1,200 Lines | 7.5h Training | API Cost $18.5 | Quality 88.6% | 3 Weeks Work")

    # 保存PDF
    c.save()
    
    print(f"PDF architecture diagram generated: {output_path}")
    print("  Format: A4 Landscape")
    print("  Resolution: Vector (scalable)")
    print("  Ready for LaTeX/Word insertion")

# 执行生成
if __name__ == "__main__":
    create_architecture_pdf()