"""Original Python-assisted task families and an observed-output shared-harness callback."""
# ruff: noqa: E701, E702
from __future__ import annotations
import csv
import hashlib
import io
import json
import math
import random
import re
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from pathlib import Path
from .generators import GENERATOR_VERSION
from .schema import SCHEMA_VERSION, canonical_json, content_hash, validate_task

TRACK_VERSION = "luna-python-original-v1"
FAMILY_SPLITS = {
 "py_math.invoice_total":"train","py_math.weighted_score":"train","py_math.batch_yield":"dev","py_math.route_cost":"test",
 "py_units.length_mix":"train","py_units.temperature_round":"train","py_units.speed_exact":"dev","py_units.volume_scale":"test",
 "py_exact.fraction_sum":"train","py_exact.ratio_compare":"train","py_exact.decimal_tax":"dev","py_exact.split_cents":"test",
 "py_count.choose":"train","py_count.multiset":"train","py_count.grid_paths":"dev","py_count.bounded_sums":"test",
 "py_stats.median_spread":"train","py_stats.harmonic_mean":"train","py_stats.correlation":"dev","py_stats.percentile_rank":"test",
 "py_table.group_totals":"train","py_table.filter_sort":"train","py_table.pivot_counts":"dev","py_table.left_join":"test",
 "py_svg.rank_bars":"train","py_svg.bucket_bars":"train","py_svg.scaled_bars":"dev","py_svg.delta_bars":"test",
 "py_docs.affine":"train","py_docs.index_origin":"train","py_docs.window_stop":"dev","py_docs.rounding_mode":"test",
}
SPLITS=("train","dev","test")
DOC_FAMILIES={x for x in FAMILY_SPLITS if x.startswith("py_docs.")}
SVG_FAMILIES={x for x in FAMILY_SPLITS if x.startswith("py_svg.")}

def _rng(family,seed):
 return random.Random(int.from_bytes(hashlib.sha256(f"{TRACK_VERSION}:{family}:{seed}".encode()).digest()[:8],"big"))
def _money(x):
 return str(x.quantize(Decimal("0.01"),rounding=ROUND_HALF_UP))
def _frac(x):
 return {"numerator":x.numerator,"denominator":x.denominator}
def _csv(headers,rows):
 s=io.StringIO(newline=""); w=csv.writer(s,lineterminator="\n"); w.writerow(headers); w.writerows(rows); return s.getvalue()

def _make(family,seed):
 r=_rng(family,seed); d={}; files={}; module=None
 if family=="py_math.invoice_total":
  d={"lines":[{"units":r.randint(1,12),"unit_price":f"{r.randint(125,980)/100:.2f}"} for _ in range(5)],"fee":f"{r.randint(125,875)/100:.2f}"}
  exp={"total":_money(sum((Decimal(x["unit_price"])*x["units"] for x in d["lines"]),Decimal(0))+Decimal(d["fee"]))}; ask="Sum units times exact decimal prices and add the fee; round HALF_UP to cents."
 elif family=="py_math.weighted_score":
  d={"scores":[{"score":f"{r.randint(500,1000)/100:.2f}","weight":r.randint(1,8)} for _ in range(5)]}; den=sum(x["weight"] for x in d["scores"]); avg=sum((Decimal(x["score"])*x["weight"] for x in d["scores"]),Decimal(0))/den
  exp={"weighted_score":str(avg.quantize(Decimal("0.001"),rounding=ROUND_HALF_UP))}; ask="Compute the weighted mean and round to three decimal places."
 elif family=="py_math.batch_yield":
  d={"packages":r.randint(3,14),"gross_grams_each":r.randint(140,800),"tare_grams_each":r.randint(5,40)}; exp={"net_grams":(d["gross_grams_each"]-d["tare_grams_each"])*d["packages"]}; ask="Subtract tare from gross for each package and report total net grams."
 elif family=="py_math.route_cost":
  d={"legs":[{"km":r.randint(12,240),"rate":f"{r.randint(15,65)/100:.2f}"} for _ in range(4)],"access_fee":f"{r.randint(250,900)/100:.2f}"}
  total=sum((Decimal(x["rate"])*x["km"] for x in d["legs"]),Decimal(0))+Decimal(d["access_fee"]); exp={"cost_credits":_money(total)}; ask="Multiply each leg's distance by its rate, add access fee, and report credits to cents."
 elif family=="py_units.length_mix":
  d={"lengths":[{"value":f"{r.randint(1,900)/10:.1f}","unit":r.choice(("mm","cm","in"))} for _ in range(6)]}; scale={"mm":Decimal(1),"cm":Decimal(10),"in":Decimal("25.4")}
  exp={"millimeters":[str((Decimal(x["value"])*scale[x["unit"]]).quantize(Decimal("0.01"),rounding=ROUND_HALF_UP)) for x in d["lengths"]]}; ask="Convert each length to millimeters in order, using 10 mm/cm and 25.4 mm/in, rounded to two places."
 elif family=="py_units.temperature_round":
  d={"celsius":[f"{r.randint(-400,405)/10:.1f}" for _ in range(7)]}; exp={"fahrenheit":[str((Decimal(x)*9/5+32).quantize(Decimal("0.1"),rounding=ROUND_HALF_UP)) for x in d["celsius"]]}; ask="Convert Celsius to Fahrenheit with Decimal and round HALF_UP to one decimal."
 elif family=="py_units.speed_exact":
  d={"km_per_hour":[r.randint(1,170) for _ in range(7)]}; exp={"meters_per_second":[_frac(Fraction(x*5,18)) for x in d["km_per_hour"]]}; ask="Convert integer km/h values exactly to m/s and return reduced numerator/denominator objects."
 elif family=="py_units.volume_scale":
  d={"measures":[{"amount":r.randint(1,15),"unit":r.choice(("ml","cup"))} for _ in range(6)]}; exp={"milliliters":[x["amount"]*(240 if x["unit"]=="cup" else 1) for x in d["measures"]]}; ask="Convert each amount to milliliters, using exactly 240 ml per cup."
 elif family=="py_exact.fraction_sum":
  d={"terms":[f"{r.randint(-12,24)}/{r.randint(2,15)}" for _ in range(7)]}; exp={"sum":_frac(sum((Fraction(x) for x in d["terms"]),Fraction(0)))}; ask="Add the rational strings exactly and return the reduced numerator and denominator."
 elif family=="py_exact.ratio_compare":
  d={"ratios":[f"{r.randint(-8,30)}/{r.randint(1,12)}" for _ in range(7)]}; i=max(range(len(d["ratios"])),key=lambda j:(Fraction(d["ratios"][j]),-j)); exp={"max_index":i,"max":_frac(Fraction(d["ratios"][i]))}; ask="Compare ratios exactly and return earliest zero-based index and reduced maximum."
 elif family=="py_exact.decimal_tax":
  d={"subtotal":f"{r.randint(299,90000)/100:.2f}","tax_rate_percent":str(r.randint(3,17))}; s=Decimal(d["subtotal"]); tax=(s*Decimal(d["tax_rate_percent"])/100).quantize(Decimal("0.01"),rounding=ROUND_HALF_UP)
  exp={"subtotal":d["subtotal"],"tax":str(tax),"total":_money(s+tax)}; ask="Calculate tax with Decimal, round tax HALF_UP to cents, then add it to subtotal."
 elif family=="py_exact.split_cents":
  n=r.randint(21,9999); d={"amount_cents":n,"recipients":[{"name":x,"weight":r.randint(1,9)} for x in ("Alder","Beryl","Cobalt","Dara")]}; den=sum(x["weight"] for x in d["recipients"]); out={x["name"]:n*x["weight"]//den for x in d["recipients"]}
  for x in sorted(d["recipients"],key=lambda x:(-(n*x["weight"]%den),x["name"]))[:n-sum(out.values())]: out[x["name"]]+=1
  exp={"shares_cents":dict(sorted(out.items()))}; ask="Allocate cents proportionally by largest remainder; break ties by recipient name."
 elif family=="py_count.choose":
  d={"n":r.randint(12,55),"r":r.randint(2,9)}; exp={"count":math.comb(d["n"],d["r"])}; ask="Count unordered groups of size r selected from n distinct items."
 elif family=="py_count.multiset":
  syms=list("AABBCDD"); r.shuffle(syms); d={"symbols":syms}; exp={"unique_arrangements":math.factorial(len(syms))//math.prod(math.factorial(syms.count(x)) for x in set(syms))}; ask="Count unique permutations of this multiset."
 elif family=="py_count.grid_paths":
  rows,cols=6,7; blocked=sorted(set((r.randrange(1,rows),r.randrange(1,cols)) for _ in range(5))); d={"rows":rows,"cols":cols,"blocked":[list(x) for x in blocked]}; dp=[[0]*cols for _ in range(rows)]
  for i in range(rows):
   for j in range(cols):
    if (i,j) not in blocked: dp[i][j]=1 if (i,j)==(0,0) else (dp[i-1][j] if i else 0)+(dp[i][j-1] if j else 0)
  exp={"paths":dp[-1][-1]}; ask="Count down/right paths from top-left to bottom-right that avoid zero-based blocked cells."
 elif family=="py_count.bounded_sums":
  d={"target":r.randint(14,28),"limits":[r.randint(4,12) for _ in range(3)]}; a,b,c=d["limits"]; t=d["target"]; exp={"solutions":sum(x+y+z==t for x in range(a+1) for y in range(b+1) for z in range(c+1))}; ask="Count ordered nonnegative triples with the target sum and given inclusive limits."
 elif family=="py_stats.median_spread":
  xs=sorted(r.randint(-50,110) for _ in range(11)); m=len(xs)//2; lo,hi=xs[:m],xs[m+1:]; d={"values":xs}; exp={"median":xs[m],"range":xs[-1]-xs[0],"iqr":hi[len(hi)//2]-lo[len(lo)//2]}; ask="Return median, range and IQR; exclude the center before taking half medians."
 elif family=="py_stats.harmonic_mean":
  d={"values":[r.randint(1,90) for _ in range(8)]}; val=Fraction(len(d["values"]),sum((Fraction(1,x) for x in d["values"]),Fraction(0))); exp={"harmonic_mean":f"{float(val):.2f}"}; ask="Compute harmonic mean of positive values and report two decimals."
 elif family=="py_stats.correlation":
  xs=[r.randint(-20,30) for _ in range(8)]; ys=[2*x+r.randint(-12,12) for x in xs]; d={"x":xs,"y":ys}; mx,my=sum(xs)/8,sum(ys)/8; n=sum((x-mx)*(y-my) for x,y in zip(xs,ys)); den=math.sqrt(sum((x-mx)**2 for x in xs)*sum((y-my)**2 for y in ys)); exp={"pearson_r":f"{n/den:.4f}" if den else "undefined"}; ask="Compute Pearson correlation for paired x/y values and round to four decimals."
 elif family=="py_stats.percentile_rank":
  xs=[r.randint(-15,50) for _ in range(10)]; target=r.randint(-15,50); n=sum(x<=target for x in xs); d={"values":xs,"target":target}; exp={"percentile_rank":_frac(Fraction(n,len(xs)))}; ask="Return the exact reduced fraction of values less than or equal to target."
 elif family=="py_table.group_totals":
  rows=[[r.choice(("juniper","linden","maple")),f"{r.randint(100,2500)/100:.2f}"] for _ in range(12)]; files["input/task.csv"]=_csv(["category","amount"],rows); d={}; totals={}
  for k,v in rows: totals[k]=totals.get(k,Decimal(0))+Decimal(v)
  exp={"totals":{k:_money(v) for k,v in sorted(totals.items())}}; ask="Read input/task.csv, group by category and sum exact decimal amounts with alphabetic keys."
 elif family=="py_table.filter_sort":
  rows=[[f"T{i:02}",r.choice(("open","closed","hold")),f"{r.randint(100,9000)/100:.2f}"] for i in range(10)]; files["input/task.csv"]=_csv(["ticket","status","amount"],rows); d={"keep_status":"open"}; files["input/task.json"]=canonical_json(d); rows2=sorted((x for x in rows if x[1]=="open"),key=lambda x:(-Decimal(x[2]),x[0]))
  exp={"rows":[{"ticket":x[0],"amount":_money(Decimal(x[2]))} for x in rows2]}; ask="Read input/task.csv, keep open rows, sort by amount descending then ticket ascending."
 elif family=="py_table.pivot_counts":
  regs=("east","north","west"); states=("done","queued","ready"); rows=[[r.choice(regs),r.choice(states)] for _ in range(18)]; files["input/task.csv"]=_csv(["region","state"],rows); d={}
  exp={"counts":{a:{b:sum(x==a and y==b for x,y in rows) for b in states} for a in regs}}; ask="Read input/task.csv and return a region-by-state count table, including zero counts and sorted keys."
 elif family=="py_table.left_join":
  products=[[f"P{i}",f"{r.randint(225,3450)/100:.2f}"] for i in range(6)]; orders=[[f"O{i}",f"P{r.randrange(8)}",r.randint(1,5)] for i in range(9)]
  files["input/products.csv"]=_csv(["product_id","unit_price"],products); files["input/task.csv"]=_csv(["order_id","product_id","quantity"],orders); prices={x:Decimal(y) for x,y in products}; d={}
  exp={"orders":[{"order_id":o,"product_id":p,"quantity":int(q),"extended":_money(prices[p]*int(q)) if p in prices else None} for o,p,q in sorted(orders)]}; ask="Read task/products CSV files, left-join on product_id, sort by order_id and return extended prices or null."
 elif family in SVG_FAMILIES:
  title={"py_svg.rank_bars":"Category frequency","py_svg.bucket_bars":"Measurement bands","py_svg.scaled_bars":"Relative levels","py_svg.delta_bars":"Absolute change"}[family]
  if family=="py_svg.rank_bars":
   vals=[r.choice(("ash","birch","cove","delta","ember")) for _ in range(15)]; d={"labels":vals,"title":title}; bars={x:vals.count(x) for x in set(vals)}; order=sorted(bars,key=lambda x:(-bars[x],x)); ask="Count input labels and write static SVG bars by descending count then label."
  elif family=="py_svg.bucket_bars":
   vals=[r.randint(1,99) for _ in range(16)]; bands=[{"name":"low","max":33},{"name":"middle","max":66},{"name":"high","max":99}]; d={"values":vals,"bands":bands,"title":title}; bars={"low":sum(x<=33 for x in vals),"middle":sum(34<=x<=66 for x in vals),"high":sum(x>=67 for x in vals)}; order=[x["name"] for x in bands]; ask="Count values in provided inclusive bands and write static SVG in band order."
  elif family=="py_svg.scaled_bars":
   vals={k:r.randint(5,150) for k in ("cedar","flint","gale","harbor")}; d={"values":vals,"title":title}; top=max(vals.values()); bars={k:round(v*100/top) for k,v in vals.items()}; order=sorted(bars); ask="Scale values to integer percent of maximum and write static SVG bars alphabetically."
  else:
   series={k:{"before":r.randint(10,90),"after":r.randint(10,90)} for k in ("north","south","west","east")}; d={"series":series,"title":title}; bars={k:abs(v["after"]-v["before"]) for k,v in series.items()}; order=sorted(bars); ask="Plot absolute before/after differences as alphabetical static SVG bars."
  exp={"title":title,"bars":bars,"order":order}
 elif family in DOC_FAMILIES:
  token=hashlib.sha256(f"{family}:{seed}:api".encode()).hexdigest()[:8]; module="local_luna_api_"+token; d={}
  if family=="py_docs.affine":
   vals=[r.randint(-18,29) for _ in range(8)]; add_first=bool(r.randrange(2)); add=r.randint(-7,9); mult=r.randint(2,6); fn="convert_"+token[:4]; sem=f"Add {add}, then multiply by {mult}." if add_first else f"Multiply by {mult}, then add {add}."; expr=f"(v+{add})*{mult}" if add_first else f"v*{mult}+{add}"
   d={"values":vals}; exp={"values":[(x+add)*mult if add_first else x*mult+add for x in vals]}; source=f'"""Local API.\nAPI callable: {fn}.\n{sem}\n"""\ndef {fn}(v):\n return {expr}\n'
  elif family=="py_docs.index_origin":
   vals=[r.randint(-18,29) for _ in range(8)]; origin=r.choice((0,1)); pos=r.randint(origin,origin+7); fn="lookup_"+token[:4]; d={"values":vals,"position":pos}; exp={"value":vals[pos-origin]}; source=f'"""Local API.\nAPI callable: {fn}.\nPublic positions are {origin}-based.\n"""\ndef {fn}(values,position):\n return values[position-{origin}]\n'
  elif family=="py_docs.window_stop":
   vals=[r.randint(-18,29) for _ in range(8)]; incl=bool(r.randrange(2)); start=r.randint(0,4); stop=start+2; fn="window_"+token[:4]; d={"values":vals,"start":start,"stop":stop}; exp={"values":vals[start:stop+int(incl)]}; word="inclusive" if incl else "exclusive"; extra=1 if incl else 0; source=f'"""Local API.\nAPI callable: {fn}.\nIndices are zero-based; stop is {word}.\n"""\ndef {fn}(values,start,stop):\n return values[start:stop+{extra}]\n'
  else:
   mode=r.choice(("ROUND_HALF_UP","ROUND_HALF_EVEN")); fn="round_"+token[:4]; value=f"{r.randint(1001,9999)/10000:.4f}"; d={"value":value,"places":2,"function":fn,"mode":mode}; rounding=ROUND_HALF_UP if mode=="ROUND_HALF_UP" else "ROUND_HALF_EVEN"; exp={"rounded":str(Decimal(value).quantize(Decimal("0.01"),rounding=rounding))}; source=f'"""Local API.\nAPI callable: {fn}.\nRounding convention: decimal.{mode}.\n"""\nfrom decimal import Decimal,{mode}\ndef {fn}(value,places):\n return str(Decimal(str(value)).quantize(Decimal(1).scaleb(-places),rounding={mode}))\n'
  files[module+".py"]=source; files["input/task.json"]=canonical_json(d); ask=f"Read python -m pydoc for {module}, discover semantics and apply its API to input/task.json."
  prompt=f"Task family: {family}; module={module}. Read input/task.json. Use Bash to read python -m pydoc for the local module, then call its discovered API from Python. {ask} Reply in one sentence: Computed result: followed by compact JSON and a period."
  return d,files,prompt,exp,module
 else: raise ValueError(f"unknown family {family}")
 if "input/task.json" not in files and not family.startswith("py_table."): files["input/task.json"]=canonical_json(d)
 if family in SVG_FAMILIES: prompt=f"Task family: {family}. Read input/task.json and compute from observed data. {ask} Write output/chart.svg then reply exactly Wrote output/chart.svg."
 else: prompt=f"Task family: {family}. Read the fixture file(s) in input/ and compute from observed data with Python. {ask} Reply in one sentence: Computed result: followed by compact JSON and a period."
 return d,files,prompt,exp,module

def generate_luna_python_task(family:str,seed:int=0)->dict:
 if family not in FAMILY_SPLITS or type(seed)is not int or seed<0: raise ValueError("invalid family or seed")
 _,files,prompt,expected,_=_make(family,seed); env={"files":files,"kv":{},"docs":[]}
 answer="Wrote output/chart.svg." if family in SVG_FAMILIES else "Computed result: "+canonical_json(expected)+"."
 oracle={"kind":"text_exact","expected":answer}
 if family in SVG_FAMILIES: oracle["artifact_path"]="output/chart.svg"; oracle["kind"]="svg"; oracle["expected"]=expected
 task={"schema_version":SCHEMA_VERSION,"task_id":f"luna-python-v1:{family}:{seed:08d}","family":family,
  "template_id":"luna-python-v1."+family,"domain":family.split(".")[0],"split":FAMILY_SPLITS[family],"seed":seed,
  "prompt":prompt,"environment":env,"oracle":oracle,"reference":{"plan":[],"final":answer},
  "provenance":{"source":"original_procedural","benchmark":False,"generator_version":GENERATOR_VERSION,
   "curriculum_track":TRACK_VERSION,"origin":"Original deterministic Python tasks; no external tasks or benchmark text"}}
 task["input_sha256"]=content_hash({"prompt":prompt,"environment":env}); validate_task(task); return task

def generate_luna_python_tasks(seeds_per_family:int=2)->list[dict]:
 if seeds_per_family<1: raise ValueError("seeds_per_family must be positive")
 return [generate_luna_python_task(f,s) for f in sorted(FAMILY_SPLITS) for s in range(seeds_per_family)]

def _code(family,module=None,docs=None):
 if family in DOC_FAMILIES:
  fn=re.search(r"API callable: ([A-Za-z_]\w*)",docs or "")
  if not fn: raise ValueError("observed docs have no callable")
  name=fn.group(1)
  if family=="py_docs.affine": result=f"{{'values':[api.{name}(x) for x in d['values']]}}"
  elif family=="py_docs.index_origin":
   if not re.search(r"positions are [01]-based",docs): raise ValueError("observed docs lack index origin")
   result=f"{{'value':api.{name}(d['values'],d['position'])}}"
  elif family=="py_docs.window_stop":
   convention=re.search(r"stop is (inclusive|exclusive)",docs)
   if not convention: raise ValueError("observed docs lack stop convention")
   end="d['stop']" if convention.group(1)=="inclusive" else "d['stop']+1"
   result=f"{{'values':api.{name}(d['values'],d['start'],{end})}}"
  else:
   if not re.search(r"Rounding convention: decimal\.ROUND_HALF_(UP|EVEN)",docs): raise ValueError("observed docs lack rounding convention")
   result=f"{{'rounded':api.{name}(d['value'],d['places'])}}"
  return f"import json,importlib,sys,os\nsys.path.insert(0,os.getcwd())\napi=importlib.import_module({module!r})\nd=json.load(open('input/task.json'))\nprint('Computed result: '+json.dumps({result},sort_keys=True,separators=(',',':'))+'.')\n"
 if family in SVG_FAMILIES:
  prep={"py_svg.rank_bars":"v=d['labels']; bars={x:v.count(x) for x in set(v)}; order=sorted(bars,key=lambda x:(-bars[x],x))",
   "py_svg.bucket_bars":"v=d['values']; b=d['bands']; bars={b[0]['name']:sum(x<=b[0]['max'] for x in v),b[1]['name']:sum(b[0]['max']<x<=b[1]['max'] for x in v),b[2]['name']:sum(x>b[1]['max'] for x in v)}; order=[x['name'] for x in b]",
   "py_svg.scaled_bars":"v=d['values']; m=max(v.values()); bars={x:round(y*100/m) for x,y in v.items()}; order=sorted(bars)",
   "py_svg.delta_bars":"bars={x:abs(y['after']-y['before']) for x,y in d['series'].items()}; order=sorted(bars)"}[family]
  return "import json\nfrom pathlib import Path\nfrom xml.sax.saxutils import escape\nd=json.load(open('input/task.json'))\n"+prep+"\nr=''.join(f'<rect data-label=\\\"{escape(x)}\\\" x=\\\"10\\\" y=\\\"{20+i*24}\\\" width=\\\"{bars[x]}\\\" height=\\\"18\\\" />' for i,x in enumerate(order))\ntitle=escape(d['title']); s=f'<svg xmlns=\\\"http://www.w3.org/2000/svg\\\" width=\\\"{max(bars.values(),default=0)+30}\\\" height=\\\"{len(bars)*24+30}\\\"><title>{title}</title>{r}</svg>'\nPath('output').mkdir(exist_ok=True); Path('output/chart.svg').write_text(s); print('Wrote output/chart.svg.')\n"
 if family.startswith("py_table."):
  scripts={
   "py_table.group_totals":"""import csv,json
from decimal import Decimal,ROUND_HALF_UP
from collections import defaultdict
a=defaultdict(Decimal)
for r in csv.DictReader(open('input/task.csv',newline='')): a[r['category']]+=Decimal(r['amount'])
o={'totals':{k:str(v.quantize(Decimal('0.01'),rounding=ROUND_HALF_UP)) for k,v in sorted(a.items())}}
print('Computed result: '+json.dumps(o,sort_keys=True,separators=(',',':'))+'.')""",
   "py_table.filter_sort":"""import csv,json
from decimal import Decimal
d=json.load(open('input/task.json')); a=[x for x in csv.DictReader(open('input/task.csv',newline='')) if x['status']==d['keep_status']]
a.sort(key=lambda x:(-Decimal(x['amount']),x['ticket']))
o={'rows':[{'ticket':x['ticket'],'amount':str(Decimal(x['amount']).quantize(Decimal('0.01')))} for x in a]}
print('Computed result: '+json.dumps(o,sort_keys=True,separators=(',',':'))+'.')""",
   "py_table.pivot_counts":"""import csv,json
a=list(csv.DictReader(open('input/task.csv',newline=''))); r=sorted({x['region'] for x in a}); s=sorted({x['state'] for x in a})
o={'counts':{x:{y:sum(z['region']==x and z['state']==y for z in a) for y in s} for x in r}}
print('Computed result: '+json.dumps(o,sort_keys=True,separators=(',',':'))+'.')""",
   "py_table.left_join":"""import csv,json
from decimal import Decimal,ROUND_HALF_UP
a=list(csv.DictReader(open('input/task.csv',newline=''))); p={x['product_id']:Decimal(x['unit_price']) for x in csv.DictReader(open('input/products.csv',newline=''))}; out=[]
for x in sorted(a,key=lambda r:r['order_id']):
 v=p.get(x['product_id']); n=int(x['quantity']); out.append({'order_id':x['order_id'],'product_id':x['product_id'],'quantity':n,'extended':str((v*n).quantize(Decimal('0.01'),rounding=ROUND_HALF_UP)) if v is not None else None})
print('Computed result: '+json.dumps({'orders':out},sort_keys=True,separators=(',',':'))+'.')"""
  }
  return scripts[family]
 return _NUMERIC[family]

_NUMERIC={
"py_math.invoice_total":"""import json
from decimal import Decimal,ROUND_HALF_UP
d=json.load(open('input/task.json')); v=sum((Decimal(x['unit_price'])*x['units'] for x in d['lines']),Decimal(0))+Decimal(d['fee'])
o={'total':str(v.quantize(Decimal('0.01'),rounding=ROUND_HALF_UP))}; print('Computed result: '+json.dumps(o,sort_keys=True,separators=(',',':'))+'.')""",
"py_math.weighted_score":"""import json
from decimal import Decimal,ROUND_HALF_UP
d=json.load(open('input/task.json')); n=sum((Decimal(x['score'])*x['weight'] for x in d['scores']),Decimal(0)); w=sum(x['weight'] for x in d['scores']); o={'weighted_score':str((n/w).quantize(Decimal('0.001'),rounding=ROUND_HALF_UP))}; print('Computed result: '+json.dumps(o,sort_keys=True,separators=(',',':'))+'.')""",
"py_math.batch_yield":"""import json
d=json.load(open('input/task.json')); o={'net_grams':(d['gross_grams_each']-d['tare_grams_each'])*d['packages']}; print('Computed result: '+json.dumps(o,sort_keys=True,separators=(',',':'))+'.')""",
"py_math.route_cost":"""import json
from decimal import Decimal,ROUND_HALF_UP
d=json.load(open('input/task.json')); v=sum((Decimal(x['rate'])*x['km'] for x in d['legs']),Decimal(0))+Decimal(d['access_fee']); o={'cost_credits':str(v.quantize(Decimal('0.01'),rounding=ROUND_HALF_UP))}; print('Computed result: '+json.dumps(o,sort_keys=True,separators=(',',':'))+'.')""",
"py_units.length_mix":"""import json
from decimal import Decimal,ROUND_HALF_UP
d=json.load(open('input/task.json')); s={'mm':Decimal(1),'cm':Decimal(10),'in':Decimal('25.4')}; o={'millimeters':[str((Decimal(x['value'])*s[x['unit']]).quantize(Decimal('0.01'),rounding=ROUND_HALF_UP)) for x in d['lengths']]}; print('Computed result: '+json.dumps(o,sort_keys=True,separators=(',',':'))+'.')""",
"py_units.temperature_round":"""import json
from decimal import Decimal,ROUND_HALF_UP
d=json.load(open('input/task.json')); o={'fahrenheit':[str((Decimal(x)*9/5+32).quantize(Decimal('0.1'),rounding=ROUND_HALF_UP)) for x in d['celsius']]}; print('Computed result: '+json.dumps(o,sort_keys=True,separators=(',',':'))+'.')""",
"py_units.speed_exact":"""import json,math
d=json.load(open('input/task.json')); a=[]
for x in d['km_per_hour']:
 n=x*5; g=math.gcd(n,18); a.append({'numerator':n//g,'denominator':18//g})
o={'meters_per_second':a}; print('Computed result: '+json.dumps(o,sort_keys=True,separators=(',',':'))+'.')""",
"py_units.volume_scale":"""import json
d=json.load(open('input/task.json')); o={'milliliters':[x['amount']*(240 if x['unit']=='cup' else 1) for x in d['measures']]}; print('Computed result: '+json.dumps(o,sort_keys=True,separators=(',',':'))+'.')""",
"py_exact.fraction_sum":"""import json
from fractions import Fraction
d=json.load(open('input/task.json')); v=sum((Fraction(x) for x in d['terms']),Fraction(0)); o={'sum':{'numerator':v.numerator,'denominator':v.denominator}}; print('Computed result: '+json.dumps(o,sort_keys=True,separators=(',',':'))+'.')""",
"py_exact.ratio_compare":"""import json
from fractions import Fraction
d=json.load(open('input/task.json')); i=max(range(len(d['ratios'])),key=lambda j:(Fraction(d['ratios'][j]),-j)); v=Fraction(d['ratios'][i]); o={'max_index':i,'max':{'numerator':v.numerator,'denominator':v.denominator}}; print('Computed result: '+json.dumps(o,sort_keys=True,separators=(',',':'))+'.')""",
"py_exact.decimal_tax":"""import json
from decimal import Decimal,ROUND_HALF_UP
d=json.load(open('input/task.json')); s=Decimal(d['subtotal']); t=(s*Decimal(d['tax_rate_percent'])/100).quantize(Decimal('0.01'),rounding=ROUND_HALF_UP); o={'subtotal':d['subtotal'],'tax':str(t),'total':str((s+t).quantize(Decimal('0.01'),rounding=ROUND_HALF_UP))}; print('Computed result: '+json.dumps(o,sort_keys=True,separators=(',',':'))+'.')""",
"py_exact.split_cents":"""import json
d=json.load(open('input/task.json')); n=d['amount_cents']; a=d['recipients']; den=sum(x['weight'] for x in a); out={x['name']:n*x['weight']//den for x in a}
for x in sorted(a,key=lambda x:(-(n*x['weight']%den),x['name']))[:n-sum(out.values())]: out[x['name']]+=1
o={'shares_cents':dict(sorted(out.items()))}; print('Computed result: '+json.dumps(o,sort_keys=True,separators=(',',':'))+'.')""",
"py_count.choose":"""import json,math
d=json.load(open('input/task.json')); o={'count':math.comb(d['n'],d['r'])}; print('Computed result: '+json.dumps(o,sort_keys=True,separators=(',',':'))+'.')""",
"py_count.multiset":"""import json,math
from collections import Counter
d=json.load(open('input/task.json')); n=math.factorial(len(d['symbols']))
for x in Counter(d['symbols']).values(): n//=math.factorial(x)
o={'unique_arrangements':n}; print('Computed result: '+json.dumps(o,sort_keys=True,separators=(',',':'))+'.')""",
"py_count.grid_paths":"""import json
d=json.load(open('input/task.json')); R,C=d['rows'],d['cols']; b={tuple(x) for x in d['blocked']}; a=[[0]*C for _ in range(R)]
for i in range(R):
 for j in range(C):
  if (i,j) not in b: a[i][j]=1 if (i,j)==(0,0) else (a[i-1][j] if i else 0)+(a[i][j-1] if j else 0)
o={'paths':a[-1][-1]}; print('Computed result: '+json.dumps(o,sort_keys=True,separators=(',',':'))+'.')""",
"py_count.bounded_sums":"""import json
d=json.load(open('input/task.json')); a,b,c=d['limits']; t=d['target']; n=sum(x+y+z==t for x in range(a+1) for y in range(b+1) for z in range(c+1)); o={'solutions':n}; print('Computed result: '+json.dumps(o,sort_keys=True,separators=(',',':'))+'.')""",
"py_stats.median_spread":"""import json,statistics
d=json.load(open('input/task.json')); a=sorted(d['values']); m=len(a)//2; lo=a[:m]; hi=a[m+1:]; o={'median':statistics.median(a),'range':a[-1]-a[0],'iqr':statistics.median(hi)-statistics.median(lo)}; print('Computed result: '+json.dumps(o,sort_keys=True,separators=(',',':'))+'.')""",
"py_stats.harmonic_mean":"""import json,statistics
d=json.load(open('input/task.json')); o={'harmonic_mean':f'{statistics.harmonic_mean(d[\"values\"]):.2f}'}; print('Computed result: '+json.dumps(o,sort_keys=True,separators=(',',':'))+'.')""",
"py_stats.correlation":"""import json,statistics
d=json.load(open('input/task.json')); o={'pearson_r':f'{statistics.correlation(d[\"x\"],d[\"y\"]):.4f}'}; print('Computed result: '+json.dumps(o,sort_keys=True,separators=(',',':'))+'.')""",
"py_stats.percentile_rank":"""import json,math
d=json.load(open('input/task.json')); n=sum(x<=d['target'] for x in d['values']); g=math.gcd(n,len(d['values'])); o={'percentile_rank':{'numerator':n//g,'denominator':len(d['values'])//g}}; print('Computed result: '+json.dumps(o,sort_keys=True,separators=(',',':'))+'.')""",
}

def _call(i,name,args):
 return {"role":"assistant","content":"","tool_calls":[{"id":i,"type":"function","function":{"name":name,"arguments":canonical_json(args)}}]}

class LunaPythonCallback:
 """Shared AgentHarness callback. It receives visible messages, never task/oracle data."""
 def __init__(self): self.n=0
 def __call__(self,messages,tools):
  prompt=next((m.get("content","") for m in messages if m.get("role")=="user"),""); family=next((f for f in FAMILY_SPLITS if prompt.startswith("Task family: "+f)),None)
  if not family: return {"role":"assistant","content":"I can't identify a supported task family from the user prompt."}
  obs=[x for x in messages if x.get("role")=="tool"]
  if family in DOC_FAMILIES and not obs:
   mod=re.search(r"module=([A-Za-z_][A-Za-z0-9_]*)",prompt)
   if not mod: return {"role":"assistant","content":"The local module name is missing."}
   self.n+=1; return _call(f"luna_py_{self.n}","bash",{"command":f"python -m pydoc {mod.group(1)}"})
  if not obs:
   self.n+=1; return _call(f"luna_py_{self.n}","python",{"code":_code(family)})
  try: result=json.loads(obs[-1]["content"])
  except (KeyError,ValueError,TypeError): return {"role":"assistant","content":"The observed tool reply was not valid JSON."}
  if "error" in result or result.get("exit_code",0)!=0 or result.get("timed_out"): return {"role":"assistant","content":"The observed tool call failed; I did not complete the task."}
  if family in DOC_FAMILIES and len(obs)==1:
   module=re.search(r"module=([A-Za-z_][A-Za-z0-9_]*)",prompt).group(1)
   try: code=_code(family,module,result.get("stdout",""))
   except ValueError as e: return {"role":"assistant","content":f"I couldn't use the observed API docs: {e}"}
   self.n+=1; return _call(f"luna_py_{self.n}","python",{"code":code})
  return {"role":"assistant","content":result.get("stdout","").strip()}

def candidate_record(task, raw_response, plan, model="gpt-6-luna", *, response_file=None, response_sha256=None):
 candidate = plan if isinstance(plan, dict) else {"actions": plan}
 return {"schema_version":"picoagent.native_candidate_plan.v1",
  "candidate_id":f"candidate:{task['task_id']}:{model}","task_id":task["task_id"],
  "task_sha256":content_hash(task),"family":task["family"],"template_id":task["template_id"],
  "split":task["split"],"status":"unexecuted_candidate_plan","execution":"unexecuted",
  "author":"luna_agent_authored_procedural_plan","model":model,
  "approach":candidate.get("approach"),"planned_tool_calls":candidate.get("actions",[]),
  "candidate":candidate,"raw_candidate_entry":raw_response,
  "raw_response_file":response_file,"raw_response_sha256":response_sha256,
  "receipts":[],"tool_events":[],"has_final_response":False,"training_eligible":False,
  "note":"Native-model-authored plan candidate only; no calls or results are observed and this is not training data."}


def materialize_luna_candidate_plans(dataset_dir):
 """Normalize preserved raw Luna replies into per-family unexecuted plan rows."""
 root=Path(dataset_dir); found=[]
 for response_path in sorted(root.glob("model_candidate*.jsonl")):
  for line in response_path.read_text(encoding="utf-8").splitlines():
   if not line: continue
   response_row=json.loads(line); raw=response_row["raw_response"]
   plans=json.loads(raw).get("candidate_plans",[])
   response_hash=hashlib.sha256(response_path.read_bytes()).hexdigest()
   for candidate in plans:
    family=candidate["family"]
    task=generate_luna_python_task(family,0)
    found.append(candidate_record(task,canonical_json(candidate),candidate,response_row["model"],
                                  response_file=response_path.name,response_sha256=response_hash))
 found.sort(key=lambda row:row["family"])
 master=root/"candidate_plans.jsonl"
 if master.exists(): master.unlink()
 _write_jsonl(master,found)
 by_split={split:[row for row in found if row["split"]==split] for split in SPLITS}
 for split,rows in by_split.items():
  path=root/f"{split}.candidate_plans.unexecuted.jsonl"
  if path.exists(): path.unlink()
  _write_jsonl(path,rows)
 counts={split:len(rows) for split,rows in by_split.items()}
 manifest={"schema":"picoagent.native_candidate_plan.manifest.v1","track":TRACK_VERSION,
  "files":{path.name:{"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"bytes":path.stat().st_size,"records":len(rows)}
           for path,rows in [(master,found)]+[(root/f"{split}.candidate_plans.unexecuted.jsonl",by_split[split]) for split in SPLITS]},
  "candidate_families":len(found),"counts_by_split":counts,"execution":"unexecuted","training_eligible":False,
  "raw_response_files":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("model_candidate*.jsonl"))}}
 path=root/"candidate_manifest.json"
 if path.exists(): path.unlink()
 path.write_text(canonical_json(manifest)+"\n",encoding="utf-8")
 main_manifest=root/"manifest.json"
 data=json.loads(main_manifest.read_text(encoding="utf-8"))
 data["candidate_count"]=len(found)
 data["candidate_artifact"]={"file":master.name,"sha256":hashlib.sha256(master.read_bytes()).hexdigest(),
                             "bytes":master.stat().st_size,"records":len(found),"execution":"unexecuted","training_eligible":False}
 main_manifest.write_text(canonical_json(data)+"\n",encoding="utf-8")
 return manifest

def _write_jsonl(path,rows):
 with Path(path).open("x",encoding="utf-8") as f:
  for row in rows: f.write(canonical_json(row)+"\n")

def write_luna_python_dataset(output_dir,seeds_per_family=2,candidates=None):
 from .generators import authored_example
 out=Path(output_dir); out.mkdir(parents=True,exist_ok=False); tasks=generate_luna_python_tasks(seeds_per_family)
 files={}
 for split in SPLITS:
  subset=[x for x in tasks if x["split"]==split]
  for kind,rows in (("tasks",subset),("authored",[authored_example(x) for x in subset])):
   name=f"{split}.{kind}.jsonl"; _write_jsonl(out/name,rows); p=out/name
   files[name]={"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"bytes":p.stat().st_size,"records":len(rows),"split":split,"kind":kind}
 candidates=candidates or {}; rows=[]
 for family,c in sorted(candidates.items()):
  if family not in FAMILY_SPLITS: raise ValueError(f"candidate for unknown family: {family}")
  task=generate_luna_python_task(family,0)
  rows.append(candidate_record(task,c["raw_response"],c["plan"],c.get("model","gpt-6-luna")))
 _write_jsonl(out/"candidate_plans.jsonl",rows); p=out/"candidate_plans.jsonl"
 candidate_artifact={"file":p.name,"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"bytes":p.stat().st_size,"records":len(rows),"execution":"unexecuted","training_eligible":False}
 manifest={"schema":"picoagent.curriculum.manifest.v1","track":TRACK_VERSION,"configuration":{"seeds_per_family":seeds_per_family,"families":len(FAMILY_SPLITS)},
  "split_policy":FAMILY_SPLITS,"split_policy_sha256":content_hash(FAMILY_SPLITS),"counts":{s:sum(x["split"]==s for x in tasks) for s in SPLITS},
  "files":files,"candidate_artifact":candidate_artifact,"execution":"unexecuted","verified_trace_count":0,"candidate_count":len(rows),"candidate_policy":{"execution":"unexecuted","training_eligible":False},
  "limitations":["Plan candidates are not tool executions or verified rollouts.","Family splits are disjoint but held-out task specifications are public.","Shared ContainerSandbox replay is required before training admission."]}
 p=out/"manifest.json"
 with p.open("x",encoding="utf-8") as f: f.write(canonical_json(manifest)+"\n")
 return p

