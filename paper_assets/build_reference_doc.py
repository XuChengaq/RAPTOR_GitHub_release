from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

OUT = Path(__file__).resolve().parent / "RAPTOR_complete_references.docx"
OUT.parent.mkdir(parents=True, exist_ok=True)

refs = [
    ("Steven L. Brunton; Bernd R. Noack; Petros Koumoutsakos", "Machine learning for fluid mechanics", "Annual Review of Fluid Mechanics, 52, 477–508 (2020)", "10.1146/annurev-fluid-010719-060214", "publisher"),
    ("Nils Thuerey; Konstantin Weißenow; Lukas Prantl; Xiangyu Hu", "Deep learning methods for Reynolds-averaged Navier–Stokes simulations of airfoil flows", "AIAA Journal, 58(1), 25–36 (2020)", "10.2514/1.J058291", "publisher"),
    ("Saakaar Bhatnagar; Yaser Afshar; Shaowu Pan; Karthik Duraisamy; Sushant Kaushik", "Prediction of aerodynamic flow fields using convolutional neural networks", "Computational Mechanics, 64, 525–545 (2019)", "10.1007/s00466-019-01740-0", "publisher"),
    ("Florent Bonnet; Jocelyn Mazari; Paola Cinnella; Patrick Gallinari", "AirfRANS: High-fidelity computational fluid dynamics dataset for approximating Reynolds-averaged Navier–Stokes solutions", "Advances in Neural Information Processing Systems 35, Datasets and Benchmarks Track (2022)", "10.52202/068431-1705", "publisher"),
    ("Mohamed Elrefaie; Florin Morar; Angela Dai; Faez Ahmed", "DrivAerNet++: A large-scale multimodal car dataset with computational fluid dynamics simulations and deep learning benchmarks", "Advances in Neural Information Processing Systems 37, Datasets and Benchmarks Track (2024)", "10.52202/079017-0016", "publisher"),
    ("Tianping Chen; Hong Chen", "Universal approximation to nonlinear operators by neural networks with arbitrary activation functions and its application to dynamical systems", "IEEE Transactions on Neural Networks, 6(4), 911–917 (1995)", "10.1109/72.392253", "publisher"),
    ("Lu Lu; Pengzhan Jin; Guofei Pang; Zhongqiang Zhang; George Em Karniadakis", "Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators", "Nature Machine Intelligence, 3, 218–229 (2021)", "10.1038/s42256-021-00302-5", "publisher"),
    ("Zongyi Li; Nikola Kovachki; Kamyar Azizzadenesheli; Burigede Liu; Kaushik Bhattacharya; Andrew M. Stuart; Anima Anandkumar", "Fourier neural operator for parametric partial differential equations", "9th International Conference on Learning Representations (ICLR) (2021)", "10.48550/arXiv.2010.08895", "version"),
    ("Nikola Kovachki; Zongyi Li; Burigede Liu; Kamyar Azizzadenesheli; Kaushik Bhattacharya; Andrew M. Stuart; Anima Anandkumar", "Neural operator: Learning maps between function spaces with applications to PDEs", "Journal of Machine Learning Research, 24(89), 1–97 (2023)", "10.48550/arXiv.2108.08481", "version"),
    ("Kamyar Azizzadenesheli; Nikola Kovachki; Zongyi Li; Miguel Liu-Schiaffini; Jean Kossaifi; Anima Anandkumar", "Neural operators for accelerating scientific simulations and design", "Nature Reviews Physics, 6, 320–328 (2024)", "10.1038/s42254-024-00712-5", "publisher"),
    ("Charles R. Qi; Hao Su; Kaichun Mo; Leonidas J. Guibas", "PointNet: Deep learning on point sets for 3D classification and segmentation", "Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 652–660 (2017)", "10.1109/CVPR.2017.16", "publisher"),
    ("William L. Hamilton; Rex Ying; Jure Leskovec", "Inductive representation learning on large graphs", "Advances in Neural Information Processing Systems 30 (2017)", "10.48550/arXiv.1706.02216", "version"),
    ("Hongyang Gao; Shuiwang Ji", "Graph U-Nets", "Proceedings of the 36th International Conference on Machine Learning, PMLR 97, 2083–2092 (2019)", "10.48550/arXiv.1905.05178", "version"),
    ("Tobias Pfaff; Meire Fortunato; Alvaro Sanchez-Gonzalez; Peter W. Battaglia", "Learning mesh-based simulation with graph networks", "9th International Conference on Learning Representations (ICLR) (2021)", "10.48550/arXiv.2010.03409", "version"),
    ("Gabriele Immordino; Andrea Vaiuso; Andrea Da Ronch; Marcello Righi", "Predicting transonic flowfields in non-homogeneous unstructured grids using autoencoder graph convolutional networks", "Journal of Computational Physics, 524, 113708 (2025)", "10.1016/j.jcp.2024.113708", "publisher"),
    ("Zongyi Li; Daniel Zhengyu Huang; Burigede Liu; Anima Anandkumar", "Fourier neural operator with learned deformations for PDEs on general geometries", "Journal of Machine Learning Research, 24(388), 1–26 (2023)", "10.48550/arXiv.2207.05209", "version"),
    ("Minglang Yin; Nicolas Charon; Ryan Brody; Lu Lu; Mauro Maggioni; Natalia Trayanova", "A scalable framework for learning the geometry-dependent solution operators of partial differential equations", "Nature Computational Science, 4, 928–940 (2024)", "10.1038/s43588-024-00732-2", "publisher"),
    ("Chenyu Zeng; Yanshu Zhang; Jiayi Zhou; Yuhan Wang; Zilin Wang; Yuhao Liu; Lei Wu; Daniel Zhengyu Huang", "Point cloud neural operator for parameterized partial differential equations on irregular geometries", "Computer Methods in Applied Mechanics and Engineering, 443, 118022 (2025)", "10.1016/j.cma.2025.118022", "publisher"),
    ("Zhongkai Hao; Zhengyi Wang; Hang Su; Chengyang Ying; Yinpeng Dong; Songming Liu; Ze Cheng; Jian Song; Jun Zhu", "GNOT: A general neural operator transformer for operator learning", "Proceedings of the 40th International Conference on Machine Learning, PMLR 202, 12556–12569 (2023)", "10.48550/arXiv.2302.14376", "version"),
    ("Haixu Wu; Huakun Luo; Haowen Wang; Jianmin Wang; Mingsheng Long", "Transolver: A fast transformer solver for PDEs on general geometries", "Proceedings of the 41st International Conference on Machine Learning, PMLR 235, 53681–53705 (2024)", "10.48550/arXiv.2402.02366", "version"),
    ("Huakun Luo; Haixu Wu; Hang Zhou; Lanxiang Xing; Yichen Di; Jianmin Wang; Mingsheng Long", "Transolver++: An accurate neural solver for PDEs on million-scale geometries", "Proceedings of the 42nd International Conference on Machine Learning, PMLR 267, 41432–41449 (2025)", "10.48550/arXiv.2502.02414", "version"),
    ("Wenjie Hu; Sidun Liu; Peng Qiao; Zhenglun Sun; Yong Dou", "Transolver is a linear transformer: Revisiting Physics-Attention through the lens of linear attention", "Proceedings of the 40th AAAI Conference on Artificial Intelligence, 408–416 (2026)", "10.1609/aaai.v40i1.37003", "publisher"),
    ("Zijiang Yang; Xiaomeng Wu; Dongmei Fu", "MoNo: Multiscale optimal transport neural operator for solving PDEs on general geometries", "arXiv preprint arXiv:2608.09764 (2026)", "10.48550/arXiv.2608.09764", "preprint"),
    ("Georgios Kissas; Jacob H. Seidman; Leonardo Ferreira Guilhoto; Victor M. Preciado; George J. Pappas; Paris Perdikaris", "Learning operators with coupled attention", "Journal of Machine Learning Research, 23(215), 1–63 (2022)", "10.48550/arXiv.2203.07404", "version"),
    ("Katiana Kontolati; Somdatta Goswami; George Em Karniadakis; Michael D. Shields", "Learning nonlinear operators in latent spaces for real-time predictions of complex dynamics in physical systems", "Nature Communications, 15, 5101 (2024)", "10.1038/s41467-024-49411-w", "publisher"),
    ("Gege Wen; Zongyi Li; Kamyar Azizzadenesheli; Anima Anandkumar; Sally M. Benson", "U-FNO—An enhanced Fourier neural operator-based deep-learning model for multiphase flow", "Advances in Water Resources, 163, 104180 (2022)", "10.1016/j.advwatres.2022.104180", "publisher"),
    ("Shizheng Wen; Arsh Kumbhat; Levi E. Lingsch; Sepehr Mousavi; Yizhou Zhao; Praveen Chandrashekar; Siddhartha Mishra", "Geometry aware operator transformer as an efficient and accurate neural surrogate for PDEs on arbitrary domains", "Advances in Neural Information Processing Systems 38 (2025)", "10.52202/085713-5197", "publisher"),
    ("Thomas Bachlechner; Bodhisattwa Prasad Majumder; Huanru Henry Mao; Gary W. Cottrell; Julian McAuley", "ReZero is all you need: Fast convergence at large depth", "Proceedings of the 37th Conference on Uncertainty in Artificial Intelligence, PMLR 161, 1352–1361 (2021)", "10.48550/arXiv.2003.04887", "version"),
    ("David Erzmann; Sören Dittmer", "Equivariant neural operators", "Journal of Computational Design and Engineering, 11(3), 91–100 (2024)", "10.1093/jcde/qwae039", "publisher"),
    ("George Em Karniadakis; Ioannis G. Kevrekidis; Lu Lu; Paris Perdikaris; Sifan Wang; Liu Yang", "Physics-informed machine learning", "Nature Reviews Physics, 3, 422–440 (2021)", "10.1038/s42254-021-00314-5", "publisher"),
    ("Unique Subedi; Ambuj Tewari", "Operator learning: A statistical perspective", "Annual Review of Statistics and Its Application, 13, 123–148 (2026)", "10.1146/annurev-statistics-042424-070908", "publisher"),
    ("Jiangce Chen; Wenzhuo Xu; Zeda Xu; Noelia Grande Gutiérrez; Sneha Prabha Narra; Christopher McComb", "Enforcing locality in neural operators for reducing generalization error", "Journal of Computational Physics, 538, 114131 (2025)", "10.1016/j.jcp.2025.114131", "publisher"),
    ("Qibang Liu; Weiheng Zhong; Hadi Meidani; Diab Abueidda; Seid Koric; Philippe H. Geubelle", "Geometry-informed neural operator transformer", "Computer Methods in Applied Mechanics and Engineering, 451, 118668 (2026)", "10.1016/j.cma.2025.118668", "publisher"),
    ("Xu-Hui Zhou; Jiequn Han; Muhammad I. Zafar; Eric M. Wolf; Christopher R. Schrock; Christopher J. Roy; Heng Xiao", "Super-fidelity neural operator: Properly learning low-fidelity data for high-fidelity prediction", "Journal of Computational Physics, 529, 113871 (2025)", "10.1016/j.jcp.2025.113871", "publisher"),
    ("Zongyi Li; Hongkai Zheng; Nikola B. Kovachki; David Jin; Haoxuan Chen; Burigede Liu; Kamyar Azizzadenesheli; Anima Anandkumar", "Physics-informed neural operator for learning partial differential equations", "ACM/IMS Journal of Data Science, 1(3), 1–27 (2024)", "10.1145/3648506", "publisher"),
]

def set_font(run, size=10, bold=False, italic=False, color="1F1F1F", name="Times New Roman"):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), name)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor.from_string(color)

def add_hyperlink(paragraph, text, url):
    part = paragraph.part
    rid = part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), rid)
    run = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    color = OxmlElement("w:color"); color.set(qn("w:val"), "1F5A94")
    underline = OxmlElement("w:u"); underline.set(qn("w:val"), "single")
    fonts = OxmlElement("w:rFonts"); fonts.set(qn("w:ascii"), "Times New Roman"); fonts.set(qn("w:hAnsi"), "Times New Roman")
    size = OxmlElement("w:sz"); size.set(qn("w:val"), "20")
    rpr.extend([fonts, size, color, underline])
    run.append(rpr)
    t = OxmlElement("w:t"); t.text = text
    run.append(t); hyperlink.append(run); paragraph._p.append(hyperlink)

def set_cell_shading(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd"); shd.set(qn("w:fill"), fill)
    tcPr.append(shd)

def set_cell_margins(cell, top=80, start=120, bottom=80, end=120):
    tc = cell._tc; tcPr = tc.get_or_add_tcPr()
    tcMar = tcPr.first_child_found_in("w:tcMar")
    if tcMar is None:
        tcMar = OxmlElement("w:tcMar"); tcPr.append(tcMar)
    for m, val in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tcMar.find(qn(f"w:{m}"))
        if node is None: node = OxmlElement(f"w:{m}"); tcMar.append(node)
        node.set(qn("w:w"), str(val)); node.set(qn("w:type"), "dxa")

def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run("Page "); set_font(run, 8.5, color="666666")
    fld = OxmlElement("w:fldSimple"); fld.set(qn("w:instr"), "PAGE")
    paragraph._p.append(fld)

doc = Document()
section = doc.sections[0]
section.page_width = Inches(8.5); section.page_height = Inches(11)
section.top_margin = Inches(0.82); section.bottom_margin = Inches(0.72)
section.left_margin = Inches(0.86); section.right_margin = Inches(0.86)
section.header_distance = Inches(0.35); section.footer_distance = Inches(0.35)

styles = doc.styles
normal = styles["Normal"]
normal.font.name = "Times New Roman"; normal.font.size = Pt(10)
normal._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
normal.paragraph_format.space_after = Pt(5)
normal.paragraph_format.line_spacing = 1.15

header = section.header.paragraphs[0]
header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
hr = header.add_run("RAPTOR manuscript  |  verified reference list")
set_font(hr, 8.5, color="6B7280")
footer = section.footer.paragraphs[0]
add_page_number(footer)

title = doc.add_paragraph()
title.paragraph_format.space_before = Pt(4); title.paragraph_format.space_after = Pt(3)
tr = title.add_run("RAPTOR Manuscript Reference List")
set_font(tr, 20, bold=True, color="17365D")
subtitle = doc.add_paragraph()
subtitle.paragraph_format.space_after = Pt(12)
sr = subtitle.add_run("35 references · complete author lists · DOI records")
set_font(sr, 10.5, italic=True, color="5B6573")

table = doc.add_table(rows=2, cols=2)
table.autofit = False
table.columns[0].width = Inches(1.35); table.columns[1].width = Inches(5.43)
data = [
    ("Scope", "References supporting the RAPTOR introduction, related work, theory, datasets, and experimental baselines."),
    ("DOI policy", "Publisher DOI is preferred. Where a venue does not assign a DOI, the DOI of the archived arXiv version is supplied and explicitly labeled as a version DOI."),
]
for i, (lab, val) in enumerate(data):
    for c in table.rows[i].cells: set_cell_margins(c)
    set_cell_shading(table.rows[i].cells[0], "E8EEF5")
    p = table.rows[i].cells[0].paragraphs[0]; p.paragraph_format.space_after = Pt(0)
    set_font(p.add_run(lab), 9.5, bold=True, color="17365D")
    p = table.rows[i].cells[1].paragraphs[0]; p.paragraph_format.space_after = Pt(0)
    set_font(p.add_run(val), 9.2, color="333333")

heading = doc.add_paragraph()
heading.paragraph_format.space_before = Pt(14); heading.paragraph_format.space_after = Pt(7)
set_font(heading.add_run("References"), 14, bold=True, color="2E5E8C")

for idx, (authors, title_text, venue, doi, doi_type) in enumerate(refs, 1):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.34)
    p.paragraph_format.first_line_indent = Inches(-0.34)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.line_spacing = 1.12
    set_font(p.add_run(f"[{idx}] "), 10, bold=True, color="17365D")
    set_font(p.add_run(authors + ". "), 10)
    set_font(p.add_run(title_text + ". "), 10, italic=True)
    set_font(p.add_run(venue + ". "), 10)
    label = {"publisher":"DOI", "version":"Version DOI", "preprint":"Preprint DOI"}[doi_type]
    set_font(p.add_run(f"{label}: "), 10, bold=True, color="333333")
    add_hyperlink(p, doi, f"https://doi.org/{doi}")

note = doc.add_paragraph()
note.paragraph_format.space_before = Pt(8); note.paragraph_format.space_after = Pt(0)
set_font(note.add_run("Verification note. "), 9, bold=True, color="17365D")
set_font(note.add_run("All author names are written out; no entry uses “et al.”. DOI labels distinguish the publisher’s version of record from an archived preprint/version DOI."), 9, color="555555")

doc.core_properties.title = "RAPTOR Manuscript Reference List"
doc.core_properties.subject = "Complete author lists and DOI records"
doc.core_properties.author = "RAPTOR research project"
doc.save(OUT)
print(OUT)
