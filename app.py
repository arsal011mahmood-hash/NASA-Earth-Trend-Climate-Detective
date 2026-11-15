"""
Earth System Trend Detective - Streamlit front end.      Run:  streamlit run app.py

Needs backend.py in the same folder, plus the Giovanni CSVs in
    data/third_pole_depletion/   data/boreal_fire_trends/   data/great_barrier_reef/
Install:  pip install streamlit pandas numpy scipy
"""
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import backend as bk

st.set_page_config(page_title="Earth System Trend Detective", page_icon="🌍", layout="wide")

STAGE_H = 800

COMPONENT_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8">
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/topojson-client@3.1.0/dist/topojson-client.min.js"></script>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{--blue:#5ec8ff;--ice:#eaf6ff;--mut:#9fb3c8;--H:__H__px}
html,body{margin:0;background:transparent;overflow:hidden;font-family:system-ui,"Segoe UI",Roboto,sans-serif;color:var(--ice)}
#stage{position:relative;width:100%;height:var(--H);overflow:hidden}
#c{position:absolute;left:0;top:0;touch-action:none;cursor:grab}
#intro{position:absolute;left:24px;top:14px;transition:opacity .8s;max-width:520px}
#intro h1{margin:0;font-size:30px}#intro p{margin:4px 0 10px;color:var(--mut)}
#chips{display:flex;flex-wrap:wrap;gap:8px}
#chips button{background:rgba(255,212,0,.1);border:1px solid rgba(255,212,0,.55);color:#ffe78a;border-radius:16px;padding:5px 13px;font-size:13px;cursor:pointer}
#chips button:hover{background:rgba(255,212,0,.25)}
#note{position:absolute;left:26px;bottom:22px;font-size:11px;color:rgba(159,179,200,.7);pointer-events:none;z-index:1}
#panel{position:absolute;left:50%;right:14px;top:12px;bottom:12px;overflow-y:auto;box-sizing:border-box;border-radius:18px;
 background:linear-gradient(160deg,rgba(14,34,64,.93),rgba(6,14,30,.97));border:1px solid rgba(150,210,255,.25);
 box-shadow:0 0 40px rgba(94,200,255,.12);opacity:0;pointer-events:none;transform:translateX(40px);transition:opacity .9s ease,transform .9s cubic-bezier(.2,.8,.2,1)}
#panel.show{opacity:1;pointer-events:auto;transform:none}
#nav{position:sticky;top:0;z-index:5;display:flex;align-items:center;gap:14px;padding:12px 22px;background:rgba(8,18,38,.95);border-bottom:1px solid rgba(150,210,255,.15)}
#crumbs{font-size:13px;color:var(--mut);flex:1}#crumbs a{color:var(--blue);cursor:pointer}#crumbs a:hover{text-decoration:underline}#crumbs i{margin:0 6px;font-style:normal}
.in{padding:6px 26px 26px}
h1.t{text-align:center;margin:10px 0 2px;font-size:28px;letter-spacing:.5px}
h2{font-size:17px;margin:22px 0 6px;color:#cfeaff;border-bottom:1px solid rgba(150,210,255,.15);padding-bottom:4px}
.sub{text-align:center;color:var(--blue);margin-bottom:12px;font-size:14px}
p{line-height:1.5;margin:8px 0}.note{color:var(--mut);font-size:13px}
button.link{background:none;border:1px solid rgba(150,210,255,.45);color:var(--ice);border-radius:20px;padding:6px 16px;cursor:pointer;font-size:13px;white-space:nowrap}
button.link:hover{background:rgba(94,200,255,.18)}
.warn{background:rgba(255,180,84,.11);border:1px solid rgba(255,180,84,.45);color:#ffd59a;border-radius:10px;padding:8px 12px;font-size:13px;margin:8px 0;line-height:1.45}
.dot{display:inline-block;width:11px;height:11px;border-radius:3px;margin:0 4px 0 10px;vertical-align:-1px}
.def{background:rgba(94,200,255,.07);border-left:3px solid var(--blue);border-radius:6px;padding:8px 12px;margin:10px 0}
.def b{color:var(--blue)}
/* ---- horizontal gradient bars ---- */
#bars{margin:10px 0 4px}
.row{display:flex;align-items:center;gap:12px;margin-bottom:9px;cursor:pointer}
.track{position:relative;flex:1;height:50px;border-radius:7px;background:rgba(255,255,255,.045);overflow:hidden;transition:box-shadow .2s}
.row:hover .track{box-shadow:0 0 0 1px rgba(160,225,255,.7),0 0 22px rgba(94,200,255,.25)}
.fill{position:absolute;left:0;top:0;bottom:0;width:0;border-right:3px solid #dff6ff;transition:width 1.2s cubic-bezier(.2,.8,.2,1)}
.fill::after{content:"";position:absolute;inset:0;background:linear-gradient(180deg,rgba(255,255,255,.2),rgba(0,0,0,.38))}
.sig .fill{background:linear-gradient(90deg,rgba(12,50,95,0) 0%,rgba(30,105,165,.6) 45%,rgba(94,200,255,.97) 100%);box-shadow:0 0 22px rgba(94,200,255,.45)}
.non .fill{background:linear-gradient(90deg,rgba(40,48,58,0) 0%,rgba(95,108,122,.6) 45%,rgba(160,174,190,.9) 100%);border-right-color:#d3dbe4}
.short .track{border:1px dashed rgba(255,180,84,.5)}
.short .fill{background:linear-gradient(90deg,rgba(60,50,30,0) 0%,rgba(140,110,60,.5) 50%,rgba(200,160,90,.75) 100%);border-right-color:#ffcf8a}
.row:hover .fill{filter:brightness(1.25)}
.name{position:absolute;left:16px;top:50%;transform:translateY(-50%);font-weight:700;font-size:15px;white-space:nowrap;z-index:2;text-shadow:0 1px 8px rgba(0,0,0,.9)}
.name em{font-size:11px;font-weight:600;color:#ffcf8a;margin-left:6px;font-style:normal}
.val{width:110px;text-align:right;font-family:ui-monospace,Consolas,monospace;font-size:13px}
.val b{display:block;font-size:15px}.val span{color:var(--mut)}
/* ---- dual range slider ---- */
.presets{display:flex;flex-wrap:wrap;gap:6px;justify-content:center;margin:12px 0 6px}
.presets button{background:rgba(255,255,255,.06);border:1px solid rgba(150,210,255,.3);color:var(--ice);border-radius:14px;padding:4px 12px;font-size:12px;cursor:pointer}
.presets button:hover,.presets button.on{background:rgba(94,200,255,.28);border-color:var(--blue)}
#wlabel{text-align:center;font-size:20px;color:var(--blue);margin:6px 0 0}
.rs{position:relative;height:70px;margin:6px 6px 0}
.rail{position:absolute;left:0;right:0;top:29px;height:4px;border-radius:2px;background:rgba(255,255,255,.15)}
#fill{position:absolute;top:0;height:100%;border-radius:2px;background:linear-gradient(90deg,#3fb6f5,#bfeaff)}
.rs input[type=range]{position:absolute;left:0;top:20px;width:100%;height:22px;margin:0;background:none;pointer-events:none;-webkit-appearance:none;appearance:none}
.rs input[type=range]::-webkit-slider-runnable-track{background:none;height:22px}
.rs input[type=range]::-moz-range-track{background:none;height:22px}
.rs input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;box-sizing:border-box;pointer-events:auto;width:22px;height:22px;border-radius:50%;background:#fff;border:4px solid #3fb6f5;box-shadow:0 0 0 4px rgba(63,182,245,.25),0 2px 8px rgba(0,0,0,.5);cursor:grab}
.rs input[type=range]::-moz-range-thumb{box-sizing:border-box;pointer-events:auto;width:22px;height:22px;border-radius:50%;background:#fff;border:4px solid #3fb6f5;box-shadow:0 0 0 4px rgba(63,182,245,.25),0 2px 8px rgba(0,0,0,.5);cursor:grab}
.bub{position:absolute;top:-4px;transform:translateX(-50%);background:#3fb6f5;color:#04121f;font-weight:700;font-size:12px;padding:2px 9px;border-radius:10px}
.tk{position:absolute;top:48px;transform:translateX(-50%);font-size:11px;color:var(--mut)}
/* ---- result cards ---- */
.card{background:rgba(255,255,255,.05);border-radius:12px;padding:12px 16px;margin-top:8px}
.chip{display:inline-block;padding:4px 12px;border-radius:14px;font-weight:600;margin:0 6px 8px 0;font-size:13px}
.chip.sig{background:#1f6fa8;color:#fff}.chip.non{background:#55616f;color:#fff}.chip.amb{background:#8a6420;color:#fff}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:6px 0}
.grid div{background:rgba(0,0,0,.25);border-radius:8px;padding:6px 10px;font-size:12px;color:var(--mut);cursor:help}
.grid b{display:block;font-size:15px;color:var(--ice);margin-top:2px}
.grid.g3{grid-template-columns:repeat(3,1fr)}
dl.gl dt{font-weight:700;color:var(--blue);margin-top:12px}dl.gl dd{margin:3px 0 0 0;line-height:1.5;color:#dbe9f6}
</style></head>
<body><div id="stage">
 <canvas id="c"></canvas>
 <div id="intro"><h1>Trend Detective</h1><p>Earth is spinning. Drag to spin it, then click one of the glowing regions (or pick one below).</p><div id="chips"></div></div>
 <div id="note"></div>
 <div id="panel">
  <div id="nav"><button class="link" id="nb">← Back to Earth</button><div id="crumbs"></div><button class="link" id="gb">Glossary</button></div>

  <div id="v-region" class="in">
   <h1 class="t" id="rt"></h1>
   <div class="sub" id="rsub"></div>
   <p id="rintro"></p>
   <div id="rwarn"></div>
   <p id="tally" style="text-align:center"></p>
   <div id="bars"></div>
   <p class="note" style="text-align:center">Longer bar = stronger trend (|Kendall τ| of the yearly means over each variable's full record). ↑ rising, ↓ falling. Dashed orange = record shorter than 10 years. Click a bar to investigate.</p>
  </div>

  <div id="v-var" class="in" style="display:none">
   <h1 class="t" id="vt"></h1>
   <div class="sub" id="vsrc"></div>
   <div id="vwarn"></div>
   <div class="def"><b>What it is.</b> <span id="vdef"></span></div>
   <div class="def"><b>Why it matters.</b> <span id="vwhy"></span></div>
   <h2>1 · Time series and trend for a chosen window</h2>
   <div class="presets" id="presets"></div>
   <div id="wlabel"></div>
   <div class="rs" id="rs"><div class="rail"><div id="fill"></div></div>
    <div class="bub" id="b0"></div><div class="bub" id="b1"></div>
    <input type="range" id="y0"><input type="range" id="y1"><div id="ticks"></div></div>
   <div id="ts"></div>
   <div class="card"><div id="verdict"></div><div class="grid" id="grid"></div><p id="sentence"></p><p class="note" id="agree"></p></div>
   <h2>2 · Extra checks on the full record</h2>
   <div id="deep"></div>
   <p class="note" style="margin-top:16px"><b>Method.</b> Each month minus that calendar month's long-term average gives the <i>anomaly</i> (this removes the seasonal cycle).
   The rate of change is a least-squares line through the monthly anomalies; its 95% confidence interval and t-test are widened for month-to-month autocorrelation.
   The Mann-Kendall test and Sen's slope use the yearly means. A straight line is only a first approximation: move the slider to see whether the trend changes over time.
   Open the <a href="#" id="gl2" style="color:var(--blue)">glossary</a> for every term.</p>
  </div>

  <div id="v-gloss" class="in" style="display:none">
   <h1 class="t">Glossary</h1><div class="sub">Every term used in this app, in plain words</div>
   <dl class="gl" id="gloss"></dl>
  </div>
 </div>
</div>
<script>
function send(type,data){window.parent.postMessage(Object.assign({isStreamlitMessage:true,type:type},data),"*");}
let started=false;
window.addEventListener("message",e=>{if(e.data&&e.data.type==="streamlit:render"&&!started){started=true;start(e.data.args.payload);}});
send("streamlit:componentReady",{apiVersion:1});
send("streamlit:setFrameHeight",{height:__H__});

const GLOSS=[
 ["Time series","A list of measurements of one quantity in time order, here one value per month, averaged over the region's box."],
 ["Area average","The mean of all satellite or model grid cells inside the box. One number per month stands for the whole region."],
 ["Anomaly","How far a month is above (+) or below (−) the normal value for that calendar month. January is compared with the average January, July with the average July. This removes the seasonal cycle so the long-term change can be seen."],
 ["Trend / rate of change","The average change per year, from the slope of a straight line fitted to the monthly anomalies by least squares. “Change over window” is that slope times the number of years."],
 ["Statistically significant","The trend is unlikely (less than 5% chance, p < 0.05) to appear just from random year-to-year ups and downs if there were really no trend. It does NOT mean the change is large or important, and it does not say what caused it."],
 ["p-value","The probability of seeing a trend at least this strong if there were really no trend. Small p (below 0.05) = strong evidence of a real trend."],
 ["95% confidence interval (CI)","The range that very likely contains the true rate of change. If the whole range is above zero (or below zero), the trend is significant at the 5% level."],
 ["t-test","The classic test of whether the fitted slope differs from zero, using the slope divided by its standard error. Here the standard error is enlarged for autocorrelation."],
 ["Autocorrelation","Neighbouring months tend to be similar (a warm month is often followed by another warm month). That means there are fewer truly independent data points than months, so the confidence interval is widened (“effective sample size”)."],
 ["Mann-Kendall test","A test that only asks whether later values tend to be higher (or lower) than earlier ones. It compares every pair of years and does not assume a straight line or a bell-shaped spread, so single extreme years matter less."],
 ["Kendall τ (tau)","The strength of the Mann-Kendall trend, from −1 (always falling) through 0 (no trend) to +1 (always rising)."],
 ["Sen's slope","The median of the slopes between every pair of years. A robust rate of change that outliers can barely move."],
 ["Seasonal Mann-Kendall","Mann-Kendall run separately for each calendar month and then combined (Hirsch-Slack method), allowing for the fact that months of the same year move together."],
 ["Hamed-Rao correction","An adjustment to the Mann-Kendall test that makes it stricter when consecutive years are correlated."],
 ["Pettitt change-point test","Looks for the year where the series most clearly jumps from one level to another (a “step” instead of a smooth trend). A small p means a step change is likely."],
 ["Drop-the-extreme-year check","The key-season test is repeated without the single most unusual year. If the trend disappears, it rested on one exceptional event rather than a steady change."],
 ["False discovery rate (FDR)","When many tests are run, a few will look significant by luck. The Benjamini-Hochberg FDR adjustment raises the p-values to keep the share of false alarms near 5%. Applied across all full-record tests in each region."],
 ["Heat-stress proxy (reef)","MMM is the warmest month of the average year. For each summer we add up how far monthly sea temperature rose above MMM (°C-months). NOAA's official alert uses daily data and degree-heating WEEKS; this is a simplified monthly indicator."],
 ["Reanalysis / land model","MERRA-2, GLDAS, FLDAS and NLDAS are computer models that combine many observations with physics to fill in gaps. They are not direct measurements, and changes in their input data can create artificial jumps."],
 ["Giovanni","NASA's online tool (giovanni.gsfc.nasa.gov) used to download these area-averaged time series."],
];

function start(D){
 const $=id=>document.getElementById(id);
 const RG=D.regions,GAP=D.min_years-1,REL=D.reliable_years,panel=$("panel"),H=__H__;
 const fmt=x=>x==null?"–":(Math.abs(x)>=1e4||(Math.abs(x)<1e-3&&x!==0)?x.toExponential(2):(+x.toPrecision(3)).toString());
 const sgn=x=>(x>0?"+":"")+fmt(x);
 const fmtP=p=>p==null?"–":p<0.001?"< 0.001":p.toFixed(3),pEq=p=>p!=null&&p<0.001?"p < 0.001":"p = "+fmtP(p);
 const stat=(v,k)=>{const s=v.stats[k];return s?{slope:s[0],lo:s[1],hi:s[2],tau:s[3],p:s[4],ny:s[5],ic:s[6],neff:s[7],tp:s[8],sen:s[9]}:null;};
 const clamp=(p,a=0,b=1)=>Math.max(a,Math.min(b,p)),lerp=(a,b,t)=>a+(b-a)*t;
 const ease=p=>p<.5?4*p*p*p:1-Math.pow(-2*p+2,3)/2;
 let view="earth",curR=null,cv=null,prevView="region";
 const G={leave:()=>{},enter:()=>{}};

 /* region geometry: a box, or (for a ring of all longitudes) four 90° pieces, each a clockwise ring */
 function ringOf(s,n,w,e){const r=[],st=.25;
  for(let y=s;y<n;y+=st)r.push([w,y]);for(let x=w;x<e;x+=st)r.push([x,n]);
  for(let y=n;y>s;y-=st)r.push([e,y]);for(let x=e;x>w;x-=st)r.push([x,s]);r.push([w,s]);return r;}
 RG.forEach((r,i)=>{const [S,N,Wd,E]=r.bbox;r.wide=(E-Wd)>170;r.phase=i*2.1;
  if(r.wide){const polys=[];for(let w=Wd;w<E-1e-6;w+=90)polys.push([ringOf(S,N,w,Math.min(E,w+90))]);
   r.geo={type:"MultiPolygon",coordinates:polys};
   const line=lat=>{const a=[];for(let x=Wd;x<=E;x+=1)a.push([x,lat]);return a;};
   r.edge={type:"MultiLineString",coordinates:[line(S),line(N)]};}
  else{r.geo={type:"Polygon",coordinates:[ringOf(S,N,Wd,E)]};r.edge=r.geo;}
  r.lab=[(Wd+E)/2,(S+N)/2];r.area=(E-Wd)*(N-S);
  r.spanRad=(r.wide?360:(E-Wd))*Math.cos(((S+N)/2)*Math.PI/180)*Math.PI/180;});
 const byArea=RG.slice().sort((a,b)=>a.area-b.area);

 /* =================== GLOBE (canvas, 60 fps) =================== */
 (function(){
  const cvs=$("c"),ctx=cvs.getContext("2d"),SPEED=6;
  let W=0,dpr=1,dirty=true,T=0,mode="spin",tw=null,hoverR=null,dragging=false,spinV=0,uvx=0,uvy=0,moved=0;
  const cam={lon:60,lat:22,R:300,s:0,tc:0};
  let L1=null,L2=null;
  const proj=d3.geoOrthographic().precision(.4),path=d3.geoPath(proj,ctx),grat=d3.geoGraticule10();
  const R0=()=>.42*Math.min(W,H);
  const R1=r=>r.wide?R0()*1.12:Math.max(900,.34*(W*.5-40)/r.spanRad);
  const mix=(a,b,t)=>d3.interpolateRgb(a,b)(t);
  const lc=(a,b,t)=>"rgba("+Math.round(lerp(a[0],b[0],t))+","+Math.round(lerp(a[1],b[1],t))+","+Math.round(lerp(a[2],b[2],t))+","+clamp(lerp(a[3],b[3],t)).toFixed(3)+")";
  const rr=(x,y,w,h,r)=>{if(ctx.roundRect)ctx.roundRect(x,y,w,h,r);else ctx.rect(x,y,w,h);};

  /* ---- satellite imagery: NASA GIBS Blue Marble tiles re-projected onto the globe (Esri World Imagery as fallback) ---- */
  const TAU=Math.PI*2,ss=p=>{p=clamp(p);return p*p*p*(p*(p*6-15)+10);};
  const SATS=[{url:"https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/BlueMarble_NextGeneration/default/GoogleMapsCompatible_Level8/{z}/{y}/{x}.jpeg",maxz:8,name:"NASA GIBS (Blue Marble)"},
   {url:"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",maxz:9,name:"Esri World Imagery"}];
  let si=0,SAT_URL=SATS[0].url,SAT_MAXZ=SATS[0].maxz,satOk=0,satBad=0,satA=0;
  const sc=document.createElement("canvas"),sg=sc.getContext("2d"),tiles=new Map();
  const setNote=()=>{$("note").textContent="Map: Natural Earth · Satellite imagery: "+SATS[si].name+" · Data: NASA Giovanni";};
  const tx=lon=>(lon+180)/360,ty=lat=>{const s=Math.sin(lat*Math.PI/180);return .5-Math.log((1+s)/(1-s))/(4*Math.PI);};
  function checkSat(){
   if(satOk>0||satBad<6)return;
   if(si<SATS.length-1){si++;SAT_URL=SATS[si].url;SAT_MAXZ=SATS[si].maxz;tiles.clear();satOk=0;satBad=0;setNote();prefetch();dirty=true;}
   else $("note").textContent="Satellite imagery could not load (internet or firewall). Showing the map outline instead.";
  }
  function getTile(z,x,y,load){
   const k=z+"/"+x+"/"+y;let t=tiles.get(k);
   if(!t&&load){t={im:new Image(),ok:false};const my=si;
    t.im.onload=()=>{t.ok=t.im.naturalWidth>0;if(t.ok&&my===si)satOk++;dirty=true;};
    t.im.onerror=()=>{t.bad=true;if(my===si){satBad++;checkSat();}};
    t.im.src=SAT_URL.replace("{z}",z).replace("{x}",x).replace("{y}",y);tiles.set(k,t);}
   return t;
  }
  function bestTile(z,x,y){
   const t=getTile(z,x,y,true);if(t&&t.ok)return{im:t.im,d:0};
   if(z>2)getTile(z-2,x>>2,y>>2,true);
   for(let d=1;d<=z;d++){const a=getTile(z-d,x>>d,y>>d,false);if(a&&a.ok)return{im:a.im,d:d};}
   return null;
  }
  function prefetch(){                                        // world at low detail + each (non-ring) region at full detail
   for(let z=0;z<=2;z++)for(let x=0;x<(1<<z);x++)for(let y=0;y<(1<<z);y++)getTile(z,x,y,true);
   RG.filter(r=>!r.wide).forEach(r=>{const bb=r.bbox,px=(bb[3]-bb[2])*.5,py=(bb[1]-bb[0])*.5;
    for(let z=3;z<=SAT_MAXZ;z++){const n=1<<z;
     for(let x=Math.floor(tx(bb[2]-px)*n);x<=Math.floor(tx(bb[3]+px)*n);x++)
      for(let y=Math.max(0,Math.floor(ty(bb[1]+py)*n));y<=Math.min(n-1,Math.floor(ty(bb[0]-py)*n));y++)getTile(z,((x%n)+n)%n,y,true);}});
  }
  function tri(im,s0x,s0y,s1x,s1y,s2x,s2y,p0,p1,p2){
   const det=(s1x-s0x)*(s2y-s0y)-(s2x-s0x)*(s1y-s0y);if(!det)return;
   const ma=((p1[0]-p0[0])*(s2y-s0y)-(p2[0]-p0[0])*(s1y-s0y))/det,mc=((s1x-s0x)*(p2[0]-p0[0])-(s2x-s0x)*(p1[0]-p0[0]))/det,
         mb=((p1[1]-p0[1])*(s2y-s0y)-(p2[1]-p0[1])*(s1y-s0y))/det,md=((s1x-s0x)*(p2[1]-p0[1])-(s2x-s0x)*(p1[1]-p0[1]))/det,
         me=p0[0]-ma*s0x-mc*s0y,mf=p0[1]-mb*s0x-md*s0y;
   const gx=(p0[0]+p1[0]+p2[0])/3,gy=(p0[1]+p1[1]+p2[1])/3;
   sg.save();sg.beginPath();
   [p0,p1,p2].forEach((p,i)=>{const dx=p[0]-gx,dy=p[1]-gy,l=Math.hypot(dx,dy)||1,x=p[0]+dx/l*.7,y=p[1]+dy/l*.7;if(i)sg.lineTo(x,y);else sg.moveTo(x,y);});
   sg.closePath();sg.clip();sg.transform(ma,mb,mc,md,me,mf);sg.drawImage(im,0,0);sg.restore();
  }
  function drawSat(Rr,b){
   let z=Math.min(SAT_MAXZ,Math.max(1,Math.ceil(Math.log2(Rr*TAU*dpr/256))));
   let dmin=1e9,dmax=-1e9,lmin=90,lmax=-90;
   for(let i=0;i<=4;i++)for(let j=0;j<=4;j++){
    const p=proj.invert([lerp(b.x0,b.x1,i/4),lerp(b.y0,b.y1,j/4)]);
    if(!p||!isFinite(p[0])||!isFinite(p[1]))return false;
    const dl=((p[0]-cam.lon+540)%360+360)%360-180;
    dmin=Math.min(dmin,dl);dmax=Math.max(dmax,dl);lmin=Math.min(lmin,p[1]);lmax=Math.max(lmax,p[1]);
   }
   if(lmax>84||lmin<-84)return false;
   let n,x0,x1,y0,y1;
   for(;;){n=1<<z;x0=Math.floor(tx(cam.lon+dmin)*n);x1=Math.floor(tx(cam.lon+dmax)*n);
    y0=Math.max(0,Math.floor(ty(lmax)*n));y1=Math.min(n-1,Math.floor(ty(lmin)*n));
    if((x1-x0+1)*(y1-y0+1)<=48||z<=1)break;z--;}
   sg.setTransform(dpr,0,0,dpr,0,0);sg.clearRect(0,0,W,H);
   const ts=Rr*TAU/n,sub=ts>360?8:ts>180?4:2,cen=[cam.lon,cam.lat];let drew=false;
   for(let xi=x0;xi<=x1;xi++)for(let yi=y0;yi<=y1;yi++){
    const xt=((xi%n)+n)%n,bt=bestTile(z,xt,yi);if(!bt)continue;
    const P=[];let bad=false,x_min=1e9,x_max=-1e9,y_min=1e9,y_max=-1e9;
    for(let j=0;j<=sub;j++)for(let i=0;i<=sub;i++){
     const lon=(xi+i/sub)/n*360-180,lat=Math.atan(Math.sinh(Math.PI*(1-2*(yi+j/sub)/n)))*57.29577951308232;
     if(d3.geoDistance([lon,lat],cen)>1.52)bad=true;
     const p=proj([lon,lat]);if(!p){bad=true;P.push([0,0]);continue;}
     P.push(p);x_min=Math.min(x_min,p[0]);x_max=Math.max(x_max,p[0]);y_min=Math.min(y_min,p[1]);y_max=Math.max(y_max,p[1]);
    }
    if(bad||x_max<b.x0||x_min>b.x1||y_max<b.y0||y_min>b.y1)continue;
    const f=1<<bt.d,isz=bt.im.naturalWidth||256,ox=xt-((xt>>bt.d)<<bt.d),oy=yi-((yi>>bt.d)<<bt.d);
    const sx=i=>(ox+i/sub)/f*isz,sy=j=>(oy+j/sub)/f*isz;
    for(let j=0;j<sub;j++)for(let i=0;i<sub;i++){
     const a=j*(sub+1)+i,c=a+sub+1;
     tri(bt.im,sx(i),sy(j),sx(i+1),sy(j),sx(i+1),sy(j+1),P[a],P[a+1],P[c+1]);
     tri(bt.im,sx(i),sy(j),sx(i+1),sy(j+1),sx(i),sy(j+1),P[a],P[c+1],P[c]);
    }
    drew=true;
   }
   return drew;
  }

  function resize(){W=$("stage").clientWidth||1000;dpr=Math.min(2,window.devicePixelRatio||1);
   cvs.width=Math.round(W*dpr);cvs.height=Math.round(H*dpr);cvs.style.width=W+"px";cvs.style.height=H+"px";sc.width=cvs.width;sc.height=cvs.height;
   if(!tw){if(mode==="spin")cam.R=R0();if(mode==="focus"&&curR)cam.R=R1(curR);}dirty=true;}
  new ResizeObserver(resize).observe($("stage"));resize();prefetch();setNote();

  const mk=w=>({land:topojson.feature(w,w.objects.land),b:topojson.mesh(w,w.objects.countries,(a,b)=>a!==b)});
  const B="https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/";
  d3.json(B+"countries-110m.json").then(w=>{L1=mk(w);dirty=true;return d3.json(B+"countries-50m.json");})
   .then(w=>{L2=mk(w);dirty=true;})
   .catch(()=>{if(!L1)$("intro").querySelector("p").textContent="Map data could not load (internet needed).";});

  function drawRegion(r){
   const foc=(r===curR),tc=foc?cam.tc:0,fade=foc?1:clamp(1-cam.tc*1.6);if(fade<=.01)return;
   const hv=(hoverR===r&&mode==="spin")?1:0,pulse=clamp(.8+.2*Math.sin(T/450+r.phase)+.2*hv,0,1);
   const b=path.bounds(r.geo);if(!isFinite(b[0][0])||!isFinite(b[1][1]))return;
   ctx.save();ctx.globalAlpha=fade;
   const gr=ctx.createLinearGradient(0,b[0][1],0,b[1][1]);
   gr.addColorStop(0,lc([255,212,0,.18+.17*hv],[245,252,255,.8*(1-.88*satA)],tc));gr.addColorStop(1,lc([255,212,0,.18+.17*hv],[80,165,235,.62*(1-.88*satA)],tc));
   ctx.beginPath();path(r.geo);ctx.fillStyle=gr;ctx.fill();ctx.lineJoin="round";
   ctx.beginPath();path(r.edge);
   ctx.strokeStyle=lc([255,212,0,.22*pulse],[170,230,255,.3],tc);ctx.lineWidth=11;ctx.stroke();
   ctx.shadowColor=lc([255,212,0,.9],[160,225,255,.9],tc);ctx.shadowBlur=16;
   ctx.strokeStyle=lc([255,212,0,pulse],[255,255,255,1],tc);ctx.lineWidth=3;ctx.stroke();ctx.shadowBlur=0;
   ctx.textAlign="center";ctx.shadowColor="rgba(0,0,0,.85)";ctx.shadowBlur=8;
   if(foc&&tc>.5){ctx.font="700 15px system-ui,sans-serif";ctx.fillStyle="rgba(255,255,255,"+((tc-.5)*2)+")";
    const lx=r.wide?W*.25:(b[0][0]+b[1][0])/2,ly=r.wide?Math.max(30,b[0][1]-14):Math.max(30,b[0][1]-14);
    ctx.fillText(r.name.toUpperCase().split("").join(" "),lx,ly);}
   else{const lp=r.wide?[cam.lon,(r.bbox[0]+r.bbox[1])/2]:r.lab;
    if(d3.geoDistance(lp,[cam.lon,cam.lat])<1.35){const p=proj(lp);
     if(p){ctx.font=(hv?"700 ":"600 ")+"13px system-ui,sans-serif";ctx.fillStyle="rgba(255,255,255,"+(1-tc*2)+")";ctx.fillText(r.name,p[0],p[1]-(r.wide?0:Math.min(40,cam.R*r.spanRad/4+12)));}}}
   ctx.restore();
  }
  function draw(){
   ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,W,H);
   const s=cam.s,tc=cam.tc,cx=lerp(W/2,W*.25,s),cy=H/2,Rd=cam.R;
   const x0=lerp(0,14,s),x1=lerp(W,W*.5-10,s),y0=lerp(0,12,s),y1=lerp(H,H-12,s),rad=lerp(0,18,s);
   proj.translate([cx,cy]).scale(Rd).rotate([-cam.lon,-cam.lat,0]).clipExtent([[-60,-60],[W+60,H+60]]);
   const zk=clamp((Rd-R0())/(R0()*1.2));
   if(zk<1){const g=ctx.createRadialGradient(cx,cy,Rd*.97,cx,cy,Rd*1.22);
    g.addColorStop(0,"rgba(120,190,255,"+(.5*(1-zk))+")");g.addColorStop(1,"rgba(120,190,255,0)");
    ctx.fillStyle=g;ctx.beginPath();ctx.arc(cx,cy,Rd*1.22,0,7);ctx.fill();}
   ctx.save();
   if(s>0){ctx.beginPath();rr(x0,y0,x1-x0,y1-y0,rad);ctx.clip();}
   const og=ctx.createRadialGradient(cx-.35*Rd,cy-.35*Rd,Rd*.1,cx,cy,Rd*1.05);
   og.addColorStop(0,mix("#3b8cf0","#14407f",tc));og.addColorStop(.6,mix("#1d5fc4","#0e3a7c",tc));og.addColorStop(1,mix("#0a2a63","#08224f",tc));
   ctx.fillStyle=og;ctx.beginPath();path({type:"Sphere"});ctx.fill();
   ctx.lineWidth=.6;ctx.strokeStyle="rgba(255,255,255,.07)";ctx.beginPath();path(grat);ctx.stroke();
   const M=(L2&&Rd>700)?L2:L1;
   if(M){ctx.fillStyle=mix("#3fa356","#3a5470",tc);ctx.beginPath();path(M.land);ctx.fill();
    ctx.strokeStyle="rgba(255,255,255,"+lerp(.55,.7,tc)+")";ctx.lineWidth=lerp(.8,1.2,tc);ctx.stroke();
    ctx.strokeStyle="rgba(255,255,255,"+lerp(.3,.45,tc)+")";ctx.lineWidth=.8;ctx.beginPath();path(M.b);ctx.stroke();}
   satA=0;
   const cover=Rd/Math.hypot((x1-x0)/2,(y1-y0)/2),sa=ss((cover-1.02)/1.4);
   if(sa>.01){let ok=false;proj.clipExtent(null);try{ok=drawSat(Rd,{x0:x0,y0:y0,x1:x1,y1:y1});}catch(err){}
    proj.clipExtent([[-60,-60],[W+60,H+60]]);
    if(ok){ctx.globalAlpha=sa;ctx.drawImage(sc,0,0,W,H);ctx.globalAlpha=1;satA=sa;
     if(M){ctx.beginPath();path(M.b);ctx.lineWidth=1;ctx.strokeStyle="rgba(255,255,255,"+(.4*sa)+")";ctx.stroke();}}}
   RG.slice().sort((a,b)=>b.area-a.area).forEach(drawRegion);   // big ring underneath, small boxes on top
   if(zk<1){const rg=ctx.createRadialGradient(cx-.2*Rd,cy-.2*Rd,Rd*.4,cx,cy,Rd);
    rg.addColorStop(0,"rgba(0,0,30,0)");rg.addColorStop(1,"rgba(0,0,30,"+(.5*(1-zk))+")");
    ctx.fillStyle=rg;ctx.beginPath();path({type:"Sphere"});ctx.fill();}
   ctx.restore();
   if(s>0){ctx.beginPath();rr(x0,y0,x1-x0,y1-y0,rad);ctx.strokeStyle="rgba(150,215,255,"+(.4*s)+")";ctx.lineWidth=1.5;ctx.stroke();}
  }

  /* ---- camera tweens ---- */
  function tween(to,dur,win,done,midAt,mid){
   mode="anim";hoverR=null;dragging=false;uvx=uvy=0;
   tw={from:Object.assign({},cam),to:to,dur:dur,win:win,done:done,t0:performance.now(),midAt:midAt,mid:mid,midDone:!mid,dl:((to.lon-cam.lon+540)%360)-180};
  }
  function stepTween(now){
   const p=clamp((now-tw.t0)/tw.dur),w=tw.win,f=tw.from,t=tw.to;
   const e=k=>ease(clamp((p-w[k][0])/(w[k][1]-w[k][0])));
   cam.lon=f.lon+tw.dl*e("rot");cam.lat=lerp(f.lat,t.lat,e("rot"));
   cam.R=Math.exp(lerp(Math.log(f.R),Math.log(t.R),e("zoom")));
   cam.s=lerp(f.s,t.s,e("slide"));cam.tc=lerp(f.tc,t.tc,e("col"));
   const it=$("intro");it.style.opacity=t.s>f.s?clamp(1-p*8):clamp((p-.8)*5);it.style.pointerEvents=t.s>f.s?"none":"auto";
   if(!tw.midDone&&p>=tw.midAt){tw.midDone=true;tw.mid();}
   if(p>=1){cam.lon=t.lon;const d=tw.done;tw=null;d();}
  }
  G.enter=function(r){
   if(mode!=="spin"||tw)return;curR=r;
   tween({lon:r.focus[0],lat:r.focus[1],R:R1(r),s:1,tc:1},4200,{rot:[0,.8],zoom:[.06,.9],slide:[.55,1],col:[.25,.85]},
    ()=>{mode="focus";dirty=true;},.8,()=>{showRegion();panel.classList.add("show");});
  };
  G.leave=function(){
   view="earth";panel.classList.remove("show");
   tween({lon:cam.lon,lat:22,R:R0(),s:0,tc:0},3400,{rot:[.3,1],zoom:[.2,1],slide:[0,.5],col:[.2,.8]},
    ()=>{mode="spin";spinV=0;curR=null;},0,null);
  };

  /* ---- interaction: drag with inertia, hover pause, click a region ---- */
  function hit(x,y){if(mode!=="spin"||tw)return null;
   if(Math.hypot(x-W/2,y-H/2)>cam.R)return null;
   const p=proj.invert([x,y]);if(!p||!isFinite(p[0]))return null;
   for(const r of byArea){if(d3.geoContains(r.geo,p))return r;
    if(!r.wide&&d3.geoDistance(p,r.lab)<14/cam.R+r.spanRad*.25)return r;}   // generous target for small boxes
   return null;}
  let lx=0,ly=0,lt=0;
  cvs.addEventListener("pointerdown",e=>{if(mode!=="spin")return;dragging=true;moved=0;lx=e.offsetX;ly=e.offsetY;lt=performance.now();uvx=uvy=0;
   try{cvs.setPointerCapture(e.pointerId);}catch(err){}cvs.style.cursor="grabbing";});
  cvs.addEventListener("pointermove",e=>{
   if(dragging){const now=performance.now(),dts=Math.max(.008,(now-lt)/1000),K=70/cam.R,dx=e.offsetX-lx,dy=e.offsetY-ly;
    cam.lon-=dx*K;cam.lat=clamp(cam.lat+dy*K,-70,70);
    uvx=clamp(.6*uvx+.4*(-dx*K/dts),-150,150);uvy=clamp(.6*uvy+.4*(dy*K/dts),-150,150);
    moved+=Math.abs(dx)+Math.abs(dy);lx=e.offsetX;ly=e.offsetY;lt=now;}
   else if(mode==="spin"){hoverR=hit(e.offsetX,e.offsetY);cvs.style.cursor=hoverR?"pointer":"grab";}});
  cvs.addEventListener("pointerup",e=>{if(!dragging)return;dragging=false;cvs.style.cursor="grab";
   if(moved<6){const r=hit(e.offsetX,e.offsetY);if(r)G.enter(r);}});
  cvs.addEventListener("pointerleave",()=>{hoverR=null;});

  let last=performance.now();
  function frame(now){
   const dt=Math.min((now-last)/1000,.05);last=now;T=now;
   if(tw)stepTween(now);
   else if(mode==="spin"){
    spinV+=(((hoverR||dragging)?0:SPEED)-spinV)*(1-Math.exp(-dt*2.2));
    cam.lon-=spinV*dt;
    if(!dragging){cam.lon+=uvx*dt;cam.lat=clamp(cam.lat+uvy*dt,-70,70);const f=Math.exp(-dt*3.2);uvx*=f;uvy*=f;
     cam.lat+=(22-cam.lat)*(1-Math.exp(-dt*.5));}
    cam.lon=((cam.lon+540)%360)-180;
   }
   if(mode!=="focus"||dirty){if(W)draw();dirty=false;}
   requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
 })();

 RG.forEach(r=>{const b=document.createElement("button");b.textContent=r.title;b.onclick=()=>G.enter(r);$("chips").appendChild(b);});

 /* =================== PANEL =================== */
 function crumb(parent,text,fn){const a=document.createElement(fn?"a":"span");a.textContent=text;if(fn)a.onclick=fn;parent.appendChild(a);}
 function sep(parent){const i=document.createElement("i");i.textContent="›";parent.appendChild(i);}
 function setNav(){
  const c=$("crumbs");c.innerHTML="";crumb(c,"Earth",()=>G.leave());sep(c);
  crumb(c,curR.title,view!=="region"?showRegion:null);
  if(view==="var"){sep(c);crumb(c,cv.label,null);}
  if(view==="gloss"){sep(c);crumb(c,"Glossary",null);}
  $("nb").textContent=view==="region"?"← Back to Earth":view==="var"?"← All variables":"← Back";
 }
 function show(id){["v-region","v-var","v-gloss"].forEach(k=>{$(k).style.display=k===id?"block":"none";});panel.scrollTop=0;}
 function back(){if(view==="var")showRegion();else if(view==="gloss"){prevView==="var"&&cv?openVar(cv):showRegion();}else if(view==="region")G.leave();}
 $("nb").onclick=back;
 document.addEventListener("keydown",e=>{if(e.key==="Escape"&&panel.classList.contains("show"))back();});
 function showGloss(){if(view!=="gloss")prevView=view;view="gloss";show("v-gloss");setNav();}
 $("gb").onclick=showGloss;$("gl2").onclick=e=>{e.preventDefault();showGloss();};
 $("gloss").innerHTML=GLOSS.map(g=>"<dt>"+g[0]+"</dt><dd>"+g[1]+"</dd>").join("");

 function showRegion(){
  view="region";show("v-region");setNav();
  const r=curR,V=r.variables;
  $("rt").textContent=r.title;
  $("rsub").textContent=r.name+" · NASA satellite and model data, "+r.years[0]+"–"+r.years[1]+" · box "+fmtBox(r.bbox);
  $("rintro").textContent=r.intro+" Below, "+V.length+" variables are tested for a trend.";
  const common=V.length>1?V[0].notes.filter(n=>V.every(v=>v.notes.includes(n))):[];   // notes shared by every variable
  $("rwarn").innerHTML=common.map(n=>'<div class="warn">⚠ '+n+"</div>").join("");
  const rows=V.map(v=>({v:v,s:stat(v,v.full)})).filter(x=>x.s);
  const rel=rows.filter(x=>!x.v.short),sh=rows.filter(x=>x.v.short);
  rel.sort((a,b)=>((a.s.p<.05?0:1)-(b.s.p<.05?0:1))||(Math.abs(b.s.tau)-Math.abs(a.s.tau)));
  const nSig=rel.filter(x=>x.s.p<.05).length;
  $("tally").innerHTML="<b>"+nSig+"</b> of "+rel.length+" variables with 10+ years show a statistically significant trend (Mann-Kendall p &lt; 0.05); <b>"+(rel.length-nSig)+"</b> do not."+
   (sh.length?"<br>"+sh.length+" more "+(sh.length>1?"have":"has")+" too short a record for a reliable test.":"")+
   '<br><span class="dot" style="background:#5ec8ff"></span>significant<span class="dot" style="background:#8794a4"></span>not significant'+(sh.length?'<span class="dot" style="background:#c8a05a"></span>short record':"");
  const box=$("bars");box.innerHTML="";
  rel.concat(sh).forEach((x,i)=>{
   const sig=x.s.p<.05,w=Math.min(100,Math.abs(x.s.tau)/.8*100),el=document.createElement("div");
   el.className="row "+(x.v.short?"short":sig?"sig":"non");
   el.innerHTML='<div class="track"><div class="fill" data-w="'+w.toFixed(1)+'" style="transition-delay:'+(i*90)+'ms"></div><div class="name">'+x.v.label+
    (x.v.short?"<em>short record ("+x.v.n_years+" yrs)</em>":"")+(x.v.notes.length>common.length&&!x.v.short?"<em>⚠ see notes</em>":"")+
    '</div></div><div class="val"><b>'+(x.s.tau>0?"↑":"↓")+" τ "+Math.abs(x.s.tau).toFixed(2)+"</b><span>p "+fmtP(x.s.p)+"</span></div>";
   el.onclick=()=>openVar(x.v);box.appendChild(el);
  });
  requestAnimationFrame(()=>requestAnimationFrame(()=>box.querySelectorAll(".fill").forEach(f=>{f.style.width=f.dataset.w+"%";})));
 }
 function fmtBox(b){const la=x=>Math.abs(x)+"°"+(x<0?"S":"N"),lo=x=>Math.abs(x)+"°"+(x<0?"W":"E");
  return (b[3]-b[2]>=300)?la(b[0])+"–"+la(b[1])+", all longitudes":la(b[0])+"–"+la(b[1])+", "+lo(b[2])+"–"+lo(b[3]);}

 /* ---- variable page + dual slider ---- */
 const s0=$("y0"),s1=$("y1");let SY0=2000,SY1=2026;
 const pos=v=>(v-SY0)/(SY1-SY0),at=f=>"calc(11px + (100% - 22px) * "+f+")";
 let pend=false;const sched=()=>{if(!pend){pend=true;requestAnimationFrame(()=>{pend=false;drawVar();});}};
 s0.oninput=()=>{if(+s0.value>+s1.value-GAP)s0.value=+s1.value-GAP;sched();};
 s1.oninput=()=>{if(+s1.value<+s0.value+GAP)s1.value=+s0.value+GAP;sched();};
 const BASEF=()=>({paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"rgba(0,0,0,0)",font:{color:"#eaf6ff",size:12}}),GR=()=>({gridcolor:"rgba(255,255,255,.1)"}),CFG={displayModeBar:false,responsive:true};

 function openVar(v){
  cv=v;view="var";show("v-var");setNav();
  $("vt").textContent=v.label+" ("+v.unit+")";$("vsrc").textContent="Data: "+v.source+" · "+v.span+" · "+v.cadence;
  $("vdef").textContent=v.definition;$("vwhy").textContent=v.why;
  $("vwarn").innerHTML=v.notes.map(n=>'<div class="warn">⚠ '+n+"</div>").join("");
  SY0=v.years[0];SY1=v.years[1];
  [s0,s1].forEach(s=>{s.min=SY0;s.max=SY1;s.step=1;});
  const tk=$("ticks");tk.innerHTML="";const step=SY1-SY0>14?5:2;
  for(let y=Math.ceil(SY0/step)*step;y<=SY1;y+=step){const t=document.createElement("div");t.className="tk";t.textContent=y;t.style.left=at(pos(y));tk.appendChild(t);}
  const pr=$("presets");pr.innerHTML="";const mid=Math.round((SY0+SY1)/2);
  [[SY0,SY1,"Full record"],[SY1-9,SY1,"Last 10 years"],[SY0,mid,"First half"],[mid,SY1,"Second half"]].forEach(p=>{
   if(p[0]<SY0||p[1]-p[0]<GAP||(p[2]!=="Full record"&&p[0]===SY0&&p[1]===SY1))return;
   const b=document.createElement("button");b.textContent=p[2];b.dataset.k=p[0]+"-"+p[1];
   b.onclick=()=>{s0.value=p[0];s1.value=p[1];sched();};pr.appendChild(b);});
  s0.value=SY0;s1.value=SY1;drawVar();drawDeep();
 }
 function drawVar(){
  const y0=+s0.value,y1=+s1.value,u=cv.unit,k=y0+"-"+y1;
  $("wlabel").textContent=y0+" – "+y1+"  ·  "+(y1-y0+1)+" years";
  $("b0").textContent=y0;$("b1").textContent=y1;$("b0").style.left=at(pos(y0));$("b1").style.left=at(pos(y1));
  $("fill").style.left=at(pos(y0));$("fill").style.width="calc((100% - 22px) * "+(pos(y1)-pos(y0))+")";
  $("presets").querySelectorAll("button").forEach(b=>b.classList.toggle("on",b.dataset.k===k));
  const s=stat(cv,k),idx=[];
  cv.dates.forEach((d,i)=>{const y=+d.slice(0,4);if(y>=y0&&y<=y1)idx.push(i);});
  if(!s||!idx.length){$("verdict").innerHTML="";$("grid").innerHTML="";$("agree").textContent="";
   $("sentence").textContent="Not enough data in this window (at least "+D.min_years+" years with 6+ months each are needed).";Plotly.purge("ts");return;}
  const X=idx.map(i=>cv.dates[i]+"-15"),A=idx.map(i=>cv.anom[i]),RAW=idx.map(i=>cv.raw[i]),by={};
  idx.forEach(i=>{const y=cv.dates[i].slice(0,4);(by[y]=by[y]||[]).push(cv.anom[i]);});
  const ay=Object.keys(by).filter(y=>by[y].length>=6),am=ay.map(y=>by[y].reduce((a,b)=>a+b,0)/by[y].length);
  const td=d=>+d.slice(0,4)+(+d.slice(5,7)-.5)/12,ends=[X[0],X[X.length-1]];
  Plotly.react("ts",[
   {x:X,y:RAW,xaxis:"x",yaxis:"y",mode:"lines",name:"Monthly value",line:{color:"#9fb3c8",width:1},hovertemplate:"%{x|%b %Y}: %{y:.4g} "+u+"<extra></extra>"},
   {x:X,y:A,xaxis:"x2",yaxis:"y2",mode:"lines",name:"Monthly anomaly",line:{color:"rgba(94,200,255,.65)",width:1.3},hovertemplate:"%{x|%b %Y}: %{y:+.3g}<extra></extra>"},
   {x:ay.map(y=>y+"-07-01"),y:am,xaxis:"x2",yaxis:"y2",mode:"markers",name:"Yearly mean anomaly",marker:{color:"#fff",size:7},hovertemplate:"%{x|%Y}: %{y:+.3g}<extra></extra>"},
   {x:ends,y:ends.map(d=>s.ic+s.slope*td(d)),xaxis:"x2",yaxis:"y2",mode:"lines",name:"Linear trend",line:{color:"#ffb454",width:3},hoverinfo:"skip"}],
   Object.assign({height:420,margin:{l:66,r:16,t:6,b:70},legend:{orientation:"h",y:-.2},hovermode:"closest",
    xaxis:Object.assign({anchor:"y",matches:"x2",showticklabels:false},GR()),
    yaxis:Object.assign({domain:[.72,1],title:{text:"value ("+u+")",font:{size:11}}},GR()),
    xaxis2:Object.assign({anchor:"y2"},GR()),
    yaxis2:Object.assign({domain:[0,.64],title:{text:"anomaly ("+u+")",font:{size:11}},zeroline:true,zerolinecolor:"rgba(255,255,255,.5)"},GR())},BASEF()),CFG);
  const sig=s.p<.05,tsig=s.tp<.05,n=y1-y0+1;
  const cell=(k,v,tip)=>'<div title="'+tip+'">'+k+"<b>"+v+"</b></div>";
  $("verdict").innerHTML='<span class="chip '+(sig?"sig":"non")+'">Mann-Kendall: '+(sig?"significant":"not significant")+'</span><span class="chip '+(tsig?"sig":"non")+'">t-test: '+(tsig?"significant":"not significant")+"</span>"+
   (cv.short?'<span class="chip amb">short record</span>':"");
  $("grid").innerHTML=
   cell("Rate of change",sgn(s.slope)+" "+u+"/yr","Least-squares slope of the monthly anomalies. Per decade: "+sgn(s.slope*10)+" "+u)+
   cell("95% confidence interval",fmt(s.lo)+" to "+fmt(s.hi),"The true rate very likely lies in this range (widened for autocorrelation). Excludes zero = significant.")+
   cell("t-test p-value",fmtP(s.tp),"Probability of a slope this steep if there were no trend, from slope ÷ standard error with "+s.neff+" effective independent months.")+
   cell("Change over window","≈ "+sgn(s.slope*n)+" "+u,"Rate of change × "+n+" years.")+
   cell("Mann-Kendall p-value",fmtP(s.p),"Rank-based trend test on the yearly mean anomalies. Below 0.05 = significant.")+
   cell("Kendall τ",(s.tau>0?"+":"")+s.tau.toFixed(2),"Trend strength from −1 (always falling) to +1 (always rising).")+
   cell("Sen's slope",sgn(s.sen)+" "+u+"/yr","Median slope between all pairs of yearly means; robust to outliers.")+
   cell("Years analysed",s.ny,"Years in the window with at least 6 months of data.");
  $("sentence").textContent=cv.label+" in the "+curR.name+" "+(s.slope>0?"increased":"decreased")+" by "+fmt(Math.abs(s.slope))+" "+u+" per year between "+y0+" and "+y1+
   " (95% CI "+fmt(s.lo)+" to "+fmt(s.hi)+"; t-test "+pEq(s.tp)+"; Mann-Kendall "+pEq(s.p)+"), so the trend is "+
   (sig&&tsig?"statistically significant":!sig&&!tsig?"not statistically significant":"uncertain")+".";
  $("agree").textContent=(sig===tsig)?"Both tests agree.":"The two tests disagree here (one is significant, the other is not), so treat this trend with caution.";
 }

 function drawDeep(){
  const v=cv,d=v.deep,el=$("deep");["season","heat","hm"].forEach(id=>{const e=$(id);if(e)Plotly.purge(e);});
  if(!d){el.innerHTML='<p class="note">These checks need at least '+REL+" complete years; this record has "+v.n_years+" years, so they are skipped.</p>";return;}
  const u=v.unit,ch=(ok,txt)=>'<span class="chip '+(ok?"sig":"non")+'">'+txt+"</span>",cell=(k,val,tip)=>'<div title="'+tip+'">'+k+"<b>"+val+"</b></div>";
  const sm=d.smk,smOK=(sm.p_fdr!=null?sm.p_fdr:sm.p)<.05;
  let h='<div class="card"><b>Seasonal Mann-Kendall</b> ('+sm.months+", "+sm.n+" complete years) "+ch(smOK,smOK?"significant after FDR":"not significant after FDR")+
   '<div class="grid">'+cell("Sen's slope",sgn(sm.slope*10)+" "+u+"/decade","Median of all year-to-year slopes, month by month.")+
   cell("95% CI",fmt(sm.lo*10)+" to "+fmt(sm.hi*10),"Per decade.")+cell("p-value",fmtP(sm.p),"Seasonal Mann-Kendall, allowing for correlation between months.")+
   cell("FDR-adjusted p",fmtP(sm.p_fdr),"Adjusted for running many tests in this region (Benjamini-Hochberg).")+"</div></div>";
  if(d.season){const q=d.season,ok=(q.p_fdr!=null?q.p_fdr:q.p)<.05,rob=q.p_without<.05;
   h+='<div class="card"><b>'+q.name+" mean</b> ("+q.n+" years, Hamed-Rao corrected) "+ch(ok,ok?"significant after FDR":"not significant after FDR")+
    '<div class="grid">'+cell("Sen's slope",sgn(q.slope*10)+" "+u+"/decade","Trend of the season's average.")+cell("p-value",fmtP(q.p),"Mann-Kendall with autocorrelation correction.")+
    cell("FDR-adjusted p",fmtP(q.p_fdr),"Adjusted for many tests.")+cell("Without "+q.extreme_year,pEq(q.p_without),"Re-tested without the single most extreme year ("+q.extreme_year+").")+"</div>"+
    '<p class="note">'+(ok?(rob?"The trend survives removing the most extreme year ("+q.extreme_year+"), so it is not driven by one event.":"Without "+q.extreme_year+" the trend is no longer significant: it leans on that one exceptional year."):"No significant trend in this season.")+'</p><div id="season"></div></div>';}
  if(v.heat){const ht=v.heat,ok=(ht.p_fdr!=null?ht.p_fdr:ht.p)<.05;
   h+='<div class="card"><b>Summer heat-stress proxy</b> (MMM = '+fmt(ht.mmm)+" °C) "+ch(ok,ok?"rising, significant":"not significant after FDR")+
    '<p class="note">Sum over Dec–Apr of how far monthly sea temperature rose above the warmest normal month (°C-months). Worst summers: '+ht.worst.join(", ")+
    ". Trend "+sgn(ht.slope*10)+" °C-months/decade, "+pEq(ht.p)+", FDR-adjusted "+pEq(ht.p_fdr)+". Monthly box averages hide local hot spots, so this understates the stress individual reefs felt.</p><div id=\"heat\"></div></div>";}
  const pt=d.pettitt,pok=pt.p<.05;
  h+='<div class="card"><b>Step change (Pettitt test)</b> '+ch(pok,pok?"likely step change":"no clear step")+
   '<p class="note">'+(pok?"The yearly means most likely jumped around <b>"+pt.year+"</b> (average shift "+sgn(pt.shift)+" "+u+", "+pEq(pt.p)+"). Part of the trend may be a sudden shift rather than a steady change"+(v.notes.some(n=>n.includes("input data"))?"; for model data, check whether this coincides with a change in the input data.":"."):
   "No convincing single jump in the yearly means (most likely year "+pt.year+", "+pEq(pt.p)+").")+"</p></div>";
  h+='<div class="card"><b>Which months drive the trend?</b><p class="note">Mann-Kendall z for each calendar month (red = rising, blue = falling, * = p &lt; 0.05). Grey = month not used.</p><div id="hm"></div></div>';
  el.innerHTML=h;
  if(d.season){const q=d.season;
   Plotly.react("season",[{type:"bar",x:q.years,y:q.values,marker:{color:q.years.map(y=>y===q.extreme_year?"#ff6b6b":"#5ec8ff")},hovertemplate:"%{x}: %{y:.4g} "+u+"<extra></extra>"},
    {x:[q.years[0],q.years[q.years.length-1]],y:[0,0].map((_,i)=>{const yy=i?q.years[q.years.length-1]:q.years[0],m=q.values.map((val,j)=>val-q.slope*q.years[j]).sort((a,b)=>a-b);return q.slope*yy+m[Math.floor(m.length/2)];}),mode:"lines",line:{color:"#ffb454",width:3},hoverinfo:"skip"}],
    Object.assign({height:230,margin:{l:60,r:10,t:6,b:30},showlegend:false,xaxis:GR(),yaxis:Object.assign({title:{text:u,font:{size:11}},range:rangeOf(q.values)},GR())},BASEF()),CFG);}
  if(v.heat){const ht=v.heat;
   Plotly.react("heat",[{type:"bar",x:ht.years,y:ht.values,marker:{color:"#ff7a59"},hovertemplate:"summer %{x}: %{y:.2f} °C-months<extra></extra>"}],
    Object.assign({height:210,margin:{l:60,r:10,t:6,b:30},showlegend:false,xaxis:GR(),yaxis:Object.assign({title:{text:"°C-months",font:{size:11}}},GR())},BASEF()),CFG);}
  const z=d.months_z.map(x=>x==null?null:x),txt=d.months_z.map((x,i)=>x==null?"":(x>0?"+":"")+x.toFixed(1)+(d.months_p[i]<.05?"*":""));
  Plotly.react("hm",[{type:"heatmap",z:[z],x:["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],y:[""],zmin:-4,zmax:4,colorscale:"RdBu",
    text:[txt],texttemplate:"%{text}",textfont:{size:12,color:"#fff"},hovertemplate:"%{x}: z = %{z:.2f}<extra></extra>",colorbar:{thickness:10,len:1,title:{text:"z",side:"right"}},xgap:2}],
   Object.assign({height:110,margin:{l:10,r:10,t:4,b:26},xaxis:{showgrid:false},yaxis:{showgrid:false,showticklabels:false},plot_bgcolor:"rgba(120,130,140,.35)"},{paper_bgcolor:"rgba(0,0,0,0)",font:{color:"#eaf6ff",size:12}}),CFG);
 }
 function rangeOf(a){const lo=Math.min(...a),hi=Math.max(...a),p=(hi-lo)*.15||1;return[lo-p,hi+p*.6];}
}
</script></body></html>
"""

# The globe + panels are one self-contained web component; it is written next to this file automatically.
_dir = Path(__file__).resolve().parent / "globe_component"
_dir.mkdir(exist_ok=True)
(_dir / "index.html").write_text(COMPONENT_HTML.replace("__H__", str(STAGE_H)), encoding="utf-8")
_comp = components.declare_component("trend_globe", path=str(_dir))

st.markdown(
    """
<style>
.stApp{background-color:#03040c;background-image:
 radial-gradient(1.5px 1.5px at 12% 22%,#fff 50%,transparent 51%),radial-gradient(1px 1px at 78% 14%,#cfe 50%,transparent 51%),
 radial-gradient(1.5px 1.5px at 35% 78%,#fff 50%,transparent 51%),radial-gradient(1px 1px at 90% 66%,#fff 50%,transparent 51%),
 radial-gradient(1px 1px at 58% 42%,#bdf 50%,transparent 51%),radial-gradient(1.5px 1.5px at 5% 88%,#fff 50%,transparent 51%);
 background-size:420px 420px;}
header[data-testid="stHeader"]{background:transparent;}
.block-container{padding:1rem 1.2rem 0 !important;max-width:100% !important;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Crunching trend statistics for every region, variable and time window...")
def load(signature):  # signature changes when you add/replace a CSV, so the cache refreshes
    return bk.build_payload()


def summary_table(payload):
    """One row per variable: full-record results, for the table and the CSV download."""
    rows = []
    for rg in payload["regions"]:
        for v in rg["variables"]:
            s = v["stats"].get(v["full"]) or [None] * 10
            d = v["deep"] or {}
            smk, pet = d.get("smk", {}), d.get("pettitt", {})
            rows.append({
                "Region": rg["title"], "Variable": v["label"], "Unit": v["unit"], "Period": v["span"],
                "Years": v["n_years"], "Rate per year (OLS)": s[0], "95% CI low": s[1], "95% CI high": s[2],
                "t-test p": s[8], "Mann-Kendall p": s[4], "Kendall tau": s[3], "Sen slope per year": s[9],
                "Seasonal MK slope per decade": None if not smk else smk["slope"] * 10,
                "Seasonal MK p (FDR)": smk.get("p_fdr"),
                "Step change year (Pettitt)": pet.get("year") if pet and pet.get("p", 1) < 0.05 else None,
                "Short record": v["short"], "Notes": " | ".join(v["notes"]),
            })
    return pd.DataFrame(rows)


payload, warnings = load(bk.data_signature())
for w in warnings:
    st.warning(w)

if not payload["regions"]:
    st.error("No data found. Put the Giovanni CSVs in data/third_pole_depletion/, data/boreal_fire_trends/ "
             "and data/great_barrier_reef/ next to app.py.")
    st.stop()

# All statistics (every slider window) are computed in backend.py and handed to the front end,
# so the slider responds instantly with no server round-trip.
_comp(payload=payload, key="main", default=None)

table = summary_table(payload)
with st.expander("📊 All results in one table (full record of each variable)"):
    st.caption("Rates are per year unless stated. 'Seasonal MK p (FDR)' is corrected for running many tests in each region. "
               "Empty Seasonal MK cells = fewer than 10 complete years.")
    st.dataframe(table.drop(columns=["Notes"]), use_container_width=True, hide_index=True)
    st.download_button("Download results as CSV", table.to_csv(index=False).encode("utf-8"),
                       "trend_results_all_regions.csv", "text/csv")
