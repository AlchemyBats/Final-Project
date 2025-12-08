#This is the manual method code. It doesn't generate the sentiment score. It only counts the positve and negative keywords.

LABEL_KEYWORDS = {
    "positive": [
        # Revenue / sales growth
        "revenue growth", "revenues grow", "revenues growing", "revenues increased",
        "sales growth", "sales grow", "sales growing", "sales increased",
        "top-line growth", "top line growth", "improving sales", "strong sales",

        # Demand / contracts
        "increased demand", "rising demand", "demand strengthened", "demand strengthening",
        "new contract", "new contracts", "contract awarded", "contracts awarded",
        "contract renewal", "contract renewals", "renewed contract", "renewed contracts",
        "backlog expansion", "expanding backlog",

        # Market expansion / opportunities
        "market expansion", "expanding market", "expansion opportunities",
        "growth opportunity", "growth opportunities",
        "new market entry", "entering new markets", "geographic expansion",

        # Regulatory tailwinds
        "regulation loosening", "regulations loosening", "regulatory loosening",
        "deregulation", "deregulated", "deregulating",
        "reduced regulatory burden", "regulatory relief",

        # Profitability
        "margin expansion", "margins expanding", "strong margins",
        "profitability improved", "improved profitability", "profits increased",
        "earnings growth", "earnings increased", "strong earnings",

        # Cost improvements
        "cost reduction", "cost reductions", "reduced costs",
        "efficiency gains", "efficiency improved", "efficiency improvements",
        "productivity gains", "productivity improved", "lower operating costs",

        # Liquidity / balance sheet strength
        "cash reserves increased", "cash position strengthened",
        "debt reduced", "reduced debt", "deleveraging",
        "improved liquidity", "strong liquidity", "improved balance sheet",

        # Operational success
        "operational improvement", "operations improved", "operational efficiencies",
        "capacity expansion", "output increased", "higher production",
        "successful integration", "successful rollout",

        # Defense budget tailwinds
        "defense budget increase", "defense spending rising", "military budget expansion",
        "higher defense appropriations", "increased pentagon funding",

        # Contract flow (defense-specific)
        "major program win", "major contract win", "large contract award",
        "multiyear contract extension", "multiyear procurement award",
        "program funding increased", "program expansion approved",

        # Backlog strength (aerospace/defense)
        "record backlog", "backlog strengthened", "backlog growth",
        "order book expansion", "order book strengthened",

        # Program performance (Lockheed-specific)
        "F-35 deliveries increased", "F-35 production ramping", "F-35 output rising",
        "missile systems demand rising", "missile orders increasing",
        "space systems growth", "satellite program growth",

        # Government/ally demand
        "international demand increasing", "foreign military sales growing",
        "ally orders improving", "global defense orders rising",

        # Cost efficiencies (industry phrasing)
        "program cost efficiencies", "program costs reduced",
        "lower manufacturing costs", "supply chain efficiencies",

        # Cash & capital structure (contextual variations)
        "strong cash generation", "cash flow improved", "higher operating cash flow",
        "balance sheet strengthening", "capital position improved",

        # Technology & R&D wins
        "successful test milestone", "successful program milestone",
        "technology demonstrator success", "R&D progress achieved",

        # Production & manufacturing improvements
        "factory throughput improved", "production throughput rising",
        "manufacturing efficiencies achieved", "cycle time reduced",

        # Workforce & hiring
        "engineering capacity expanded", "workforce expanded",
        "staffing improved", "hiring improved",

        # Regulatory / political support (defense-specific)
        "bipartisan defense support", "congressional support increased",
        "program fully funded", "appropriations approved"

    ],

    "negative": [
        # Revenue / sales decline
        "revenue decline", "revenues declined", "revenues decreased", "revenues decreasing",
        "sales decline", "sales declined", "sales decreased", "sales decreasing",
        "top-line decline", "top line decline", "weak sales",

        # Demand weakness
        "declining demand", "weak demand", "demand decreased", "demand softening",
        "reduced orders", "order cancellation", "order cancellations",

        # Cost pressures
        "cost increase", "cost increases", "rising costs", "increased costs",
        "cost pressure", "cost pressures", "higher operating costs",
        "inflationary pressure", "input cost inflation",

        # Supply chain issues
        "supply chain disruption", "supply chain disruptions",
        "supply disruption", "supply disruptions",
        "logistics delays", "production delays",
        "material shortage", "component shortage",

        # Regulatory headwinds
        "new regulation", "new regulations",
        "tightened regulation", "tightened regulations",
        "increased regulation", "regulatory burden", "regulatory burdens",
        "compliance costs", "compliance burden",

        # Profitability deterioration
        "margin compression", "compressed margins",
        "profitability declined", "declining profitability", "profits decreased",
        "earnings decline", "earnings decreased", "reduced profitability",

        # Operational issues
        "operational challenge", "operational challenges",
        "operational disruption", "operational disruptions",
        "production issue", "production issues",
        "capacity constraint", "capacity constraints",

        # Financial risk
        "debt increased", "increased debt", "leveraged position",
        "liquidity risk", "cash decreased", "reduced liquidity",
        "higher interest expense", "credit risk",

        # Defense budget risk
        "defense budget cuts", "reduced defense funding",
        "appropriations delayed", "continuing resolution impact",
        "government funding uncertainty",

        # Contract/program issues
        "contract loss", "lost contract", "contract not renewed",
        "program delay", "program delays", "program setback",
        "program cost increase", "program overruns", "cost overrun",

        # Backlog or order weakness
        "order book decline", "backlog reduction", "reduced backlog",

        # Supply chain (defense-specific)
        "titanium shortage", "critical materials shortage",
        "supplier delay", "supplier setback", "supplier quality issue",

        # F-35 / LM-specific problems
        "F-35 delivery delays", "F-35 production issues",
        "airframe issues", "quality escape", "rework required",
        "missile production delays", "satellite launch delay",

        # Regulatory/policy pressure
        "export restriction", "export limitations", "ITAR restriction",
        "license delays", "compliance investigation",

        # Profit and cost risk (industry phrasing)
        "fixed-price contract pressure", "fixed-price exposure",
        "development cost escalation", "program margin pressure",

        # Workforce issues
        "labor shortage", "labor constraints", "engineering shortage",
        "hiring challenges", "clearance delays",

        # Political/geopolitical uncertainty
        "arms sale blocked", "arms sale delayed",
        "foreign approval delayed", "ally procurement postponed",

        # Operational disruptions (expanded variants)
        "manufacturing halt", "production stoppage",
        "factory downtime", "supply bottleneck",
        "schedule slip", "schedule slippage"

    ],

    "neutral": [
        # Routine reporting language
        "in line with expectations", "as expected", "consistent with prior period",
        "no material change", "no significant change", "unchanged results",
        "expected results", "typical results",

        # Standard operations
        "ongoing operations", "routine operations", "normal operations",
        "regular activities", "standard business operations",
        "standard procedures", "normal course operations",

        # Regulatory neutrality
        "regulatory compliance", "compliant with regulations",
        "compliance maintained", "standard compliance practices",

        # Market stability
        "stable demand", "stable pricing", "unchanged market conditions",
        "flat market conditions", "steady market environment",

        # Financial stability
        "steady cash flow", "consistent cash flow",
        "unchanged debt", "stable liquidity",
        "neutral cash movement", "neutral cash flow",

        # Operational stability
        "capacity maintained", "operations maintained",
        "standard production", "typical production levels",
        "normal utilization", "steady utilization",

        # General neutral descriptors
        "steady performance", "consistent performance",
        "typical variation", "expected variation",
        "standard operating rhythm", "unchanged operations",

        # Budget and government process (neutral tone)
        "awaiting appropriations", "pending budget resolution",
        "standard procurement timeline", "routine budget cycle",

        # Typical defense reporting language
        "program progressing as planned", "program on track",
        "program in steady state", "contract performance as expected",

        # Backlog / order routine
        "stable backlog", "backlog unchanged",
        "order flow steady", "steady order intake",

        # Operational stability (aerospace phrasing)
        "stable production rate", "production cadence maintained",
        "manufacturing output steady", "factory utilization steady",

        # Program / milestone neutrality
        "routine milestone", "standard testing milestone",
        "expected test schedule", "normal development cadence",

        # Workforce / hiring neutrality
        "staffing stable", "workforce stable", "headcount unchanged",

        # Regulatory neutrality (defense context)
        "standard ITAR compliance", "routine compliance activity",
        "no material compliance events",

        # International sales routine
        "typical foreign military sales", "routine international activity",
        "stable global demand"

    ]
}






import os
import csv

# Ensure output folder exists
os.makedirs("manual_summaries", exist_ok=True)

input_folder = "reports_formatted"
output_csv = "manual_summaries/results.csv"

results = []

# Loop through each report file
for fname in os.listdir(input_folder):
    if not fname.lower().startswith("lmt_") or not fname.lower().endswith(".txt"):
        continue

    # extract date: lmt_form_yy-mm-dd.txt → last part before .txt
    try:
        date_part = fname.rsplit("_", 1)[1].replace(".txt", "")
    except Exception:
        continue

    # read file
    try:
        with open(os.path.join(input_folder, fname), "r", encoding="utf-8") as file:
            raw = file.read().lower()
    except FileNotFoundError:
        print(f"Missing file: {fname}")
        continue
    except Exception as e:
        print(f"Error reading {fname}: {str(e)}")
        continue

    # keyword counting
    counts = {category: 0 for category in LABEL_KEYWORDS}

    for category, phrases in LABEL_KEYWORDS.items():
        for phrase in phrases:
            counts[category] += raw.count(phrase)

    # store results
    results.append([
        date_part,
        counts.get("positive", 0),
        counts.get("negative", 0),
        counts.get("neutral", 0)
    ])

# write CSV
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Date", "positives", "negatives", "neutral"])
    writer.writerows(results)

print("Saved: manual_summaries/results.csv")

