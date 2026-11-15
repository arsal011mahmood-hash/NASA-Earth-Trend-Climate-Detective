"""
Earth System Trend Detective - Streamlit front end.   Run:  streamlit run app.py

Needs backend.py in the same folder, plus CSVs in data/karakoram/ (see backend.py).
Install:  pip install streamlit pandas numpy scipy
"""
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

import backend as bk

st.set_page_config(page_title="Earth System Trend Detective", page_icon="🌍", layout="wide")

COMPONENT_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8">
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/topojson-client@3.1.0/dist/topojson-client.min.js"></script>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{--blue:#5ec8ff;--ice:#eaf6ff;--mut:#9fb3c8}
html,body{margin:0;background:transparent;overflow:hidden;font-family:system-ui,"Segoe UI",Roboto,sans-serif;color:var(--ice)}
#stage{position:relative;width:100%;height:760px;overflow:hidden}
#c{position:absolute;left:0;top:0;touch-action:none;cursor:grab}
#intro{position:absolute;left:24px;top:14px;pointer-events:none;transition:opacity .8s}
#intro h1{margin:0;font-size:30px}#intro p{margin:4px 0 0;color:var(--mut)}
#note{position:absolute;left:26px;bottom:22px;font-size:11px;color:rgba(159,179,200,.7);pointer-events:none;z-index:1}
#panel{position:absolute;left:50%;right:14px;top:12px;bottom:12px;overflow-y:auto;box-sizing:border-box;border-radius:18px;
 background:linear-gradient(160deg,rgba(14,34,64,.92),rgba(6,14,30,.96));border:1px solid rgba(150,210,255,.25);
 box-shadow:0 0 40px rgba(94,200,255,.12);opacity:0;pointer-events:none;transform:translateX(40px);transition:opacity .9s ease,transform .9s cubic-bezier(.2,.8,.2,1)}
#panel.show{opacity:1;pointer-events:auto;transform:none}
#nav{position:sticky;top:0;z-index:5;display:flex;align-items:center;gap:14px;padding:12px 22px;background:rgba(8,18,38,.94);border-bottom:1px solid rgba(150,210,255,.15)}
#crumbs{font-size:13px;color:var(--mut)}#crumbs a{color:var(--blue);cursor:pointer}#crumbs a:hover{text-decoration:underline}#crumbs i{margin:0 6px;font-style:normal}
.in{padding:6px 26px 26px}
h1.t{text-align:center;margin:10px 0 2px;font-size:30px;letter-spacing:.5px}
.sub{text-align:center;color:var(--blue);margin-bottom:12px}
p{line-height:1.5;margin:8px 0}.note{color:var(--mut);font-size:13px}
button.link{background:none;border:1px solid rgba(150,210,255,.45);color:var(--ice);border-radius:20px;padding:6px 16px;cursor:pointer;font-size:13px}
button.link:hover{background:rgba(94,200,255,.18)}
.warn{background:rgba(255,180,84,.12);border:1px solid rgba(255,180,84,.5);color:#ffcf8a;border-radius:10px;padding:8px 12px;font-size:13px;margin:8px 0}
.dot{display:inline-block;width:11px;height:11px;border-radius:3px;margin:0 4px 0 10px;vertical-align:-1px}
/* ---- horizontal gradient bars ---- */
#bars{margin:10px 0 4px}
.row{display:flex;align-items:center;gap:12px;margin-bottom:9px;cursor:pointer}
.track{position:relative;flex:1;height:50px;border-radius:7px;background:rgba(255,255,255,.045);overflow:hidden;transition:box-shadow .2s}
.row:hover .track{box-shadow:0 0 0 1px rgba(160,225,255,.7),0 0 22px rgba(94,200,255,.25)}
.fill{position:absolute;left:0;top:0;bottom:0;width:0;border-right:3px solid #dff6ff;transition:width 1.2s cubic-bezier(.2,.8,.2,1)}
.fill::after{content:"";position:absolute;inset:0;background:linear-gradient(180deg,rgba(255,255,255,.2),rgba(0,0,0,.38))}
.sig .fill{background:linear-gradient(90deg,rgba(12,50,95,0) 0%,rgba(30,105,165,.6) 45%,rgba(94,200,255,.97) 100%);box-shadow:0 0 22px rgba(94,200,255,.45)}
.non .fill{background:linear-gradient(90deg,rgba(40,48,58,0) 0%,rgba(95,108,122,.6) 45%,rgba(160,174,190,.9) 100%);border-right-color:#d3dbe4}
.row:hover .fill{filter:brightness(1.25)}
.name{position:absolute;left:16px;top:50%;transform:translateY(-50%);font-weight:700;font-size:16px;white-space:nowrap;z-index:2;text-shadow:0 1px 8px rgba(0,0,0,.9)}
.name em{font-size:11px;font-weight:600;color:#ffcf8a;margin-left:6px}
.val{width:104px;text-align:right;font-family:ui-monospace,Consolas,monospace;font-size:13px}
.val b{display:block;font-size:15px}.val span{color:var(--mut)}
/* ---- dual range slider ---- */
.presets{display:flex;flex-wrap:wrap;gap:6px;justify-content:center;margin:6px 0}
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
/* ---- result card ---- */
#card{background:rgba(255,255,255,.05);border-radius:12px;padding:12px 16px;margin-top:6px}
.chip{display:inline-block;padding:4px 12px;border-radius:14px;font-weight:600;margin-bottom:8px}
.chip.sig{background:#1f6fa8;color:#fff}.chip.non{background:#55616f;color:#fff}
#grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:6px 0}
#grid div{background:rgba(0,0,0,.25);border-radius:8px;padding:6px 10px;font-size:12px;color:var(--mut)}
#grid b{display:block;font-size:15px;color:var(--ice)}
</style></head>
<body><div id="stage">
 <canvas id="c"></canvas>
 <div id="intro"><h1>Trend Detective</h1><p>Earth is spinning. Drag to spin it yourself, then click the glowing region.</p></div>
 <div id="note"></div>
 <div id="panel">
  <div id="nav"><button class="link" id="nb">← Back to Earth</button><div id="crumbs"></div></div>
  <div id="v-region" class="in">
   <h1 class="t">Third Pole Depletion</h1>
   <div class="sub" id="rsub"></div>
   <p id="tpdef"></p>
   <div id="demo-warn"></div>
   <p id="tally" style="text-align:center"></p>
   <div id="bars"></div>
   <p class="note" style="text-align:center">Longer bar = stronger trend (|Kendall τ|). ↑ rising, ↓ falling. Click a bar to investigate.</p>
  </div>
  <div id="v-var" class="in" style="display:none">
   <h1 class="t" id="vt"></h1>
   <div class="sub" id="vsrc"></div>
   <div id="vdemo"></div>
   <p id="vdef"></p><p class="note" id="vwhy"></p>
   <div class="presets" id="presets"></div>
   <div id="wlabel"></div>
   <div class="rs" id="rs"><div class="rail"><div id="fill"></div></div>
    <div class="bub" id="b0"></div><div class="bub" id="b1"></div>
    <input type="range" id="y0"><input type="range" id="y1"><div id="ticks"></div></div>
   <div id="ts"></div>
   <div id="card"><div id="verdict"></div><div id="grid"></div><p id="sentence"></p><p class="note" id="agree"></p></div>
   <p class="note">Method: each month minus that calendar month's long-term average (removes the seasonal cycle). Linear trend by least squares, with a confidence interval widened for month-to-month autocorrelation. Mann-Kendall test on yearly averages. A straight line is only a first approximation: move the slider to see whether the trend changes over time.</p>
  </div>
 </div>
</div>
<script>
function send(type,data){window.parent.postMessage(Object.assign({isStreamlitMessage:true,type:type},data),"*");}
let started=false;
window.addEventListener("message",e=>{if(e.data&&e.data.type==="streamlit:render"&&!started){started=true;start(e.data.args.payload);}});
send("streamlit:componentReady",{apiVersion:1});
send("streamlit:setFrameHeight",{height:760});

function start(D){
 const $=id=>document.getElementById(id);
 const R=D.region,bb=R.bbox,clon=(bb[2]+bb[3])/2,clat=(bb[0]+bb[1])/2;
 const V=D.variables,Y0=D.years[0],Y1=D.years[1],FULL=Y0+"-"+Y1,panel=$("panel");
 const fmt=x=>(+x.toPrecision(3)).toString(),fmtP=p=>p<0.001?"<0.001":p.toFixed(3);
 const stat=(v,k)=>{const s=v.stats[k];return s?{slope:s[0],lo:s[1],hi:s[2],tau:s[3],p:s[4],ny:s[5],ic:s[6]}:null;};
 const clamp=(p,a=0,b=1)=>Math.max(a,Math.min(b,p)),lerp=(a,b,t)=>a+(b-a)*t;
 const ease=p=>p<.5?4*p*p*p:1-Math.pow(-2*p+2,3)/2;
 let view="earth",cv=null;
 const G={leave:()=>{}};

 /* =================== GLOBE (canvas, 60 fps) =================== */
 (function(){
  const cvs=$("c"),ctx=cvs.getContext("2d"),SPEED=6,H=760;
  let W=0,dpr=1,dirty=true,T=0,mode="spin",tw=null,hover=false,dragging=false,spinV=0,uvx=0,uvy=0,moved=0;
  const cam={lon:60,lat:22,R:300,s:0,tc:0};
  let L1=null,L2=null;
  const proj=d3.geoOrthographic().precision(.4),path=d3.geoPath(proj,ctx),grat=d3.geoGraticule10();
  // clockwise ring, densified (counter-clockwise would select the whole sphere)
  const ring=[];(function(){const s=bb[0],n=bb[1],w=bb[2],e=bb[3],st=.25;
   for(let y=s;y<n;y+=st)ring.push([w,y]);for(let x=w;x<e;x+=st)ring.push([x,n]);
   for(let y=n;y>s;y-=st)ring.push([e,y]);for(let x=e;x>w;x-=st)ring.push([x,s]);ring.push([w,s]);})();
  const regionGeo={type:"Polygon",coordinates:[ring]};
  const spanRad=(bb[3]-bb[2])*Math.cos(clat*Math.PI/180)*Math.PI/180;
  const R0=()=>.42*Math.min(W,H),R1=()=>Math.max(900,.34*(W*.5-40)/spanRad);
  const mix=(a,b,t)=>d3.interpolateRgb(a,b)(t);
  const lc=(a,b,t)=>"rgba("+Math.round(lerp(a[0],b[0],t))+","+Math.round(lerp(a[1],b[1],t))+","+Math.round(lerp(a[2],b[2],t))+","+clamp(lerp(a[3],b[3],t)).toFixed(3)+")";
  const rr=(x,y,w,h,r)=>{if(ctx.roundRect)ctx.roundRect(x,y,w,h,r);else ctx.rect(x,y,w,h);};

  /* ---- satellite imagery: NASA GIBS Blue Marble tiles re-projected onto the globe (Esri World Imagery is the automatic fallback) ---- */
  const TAU=Math.PI*2,ss=p=>{p=clamp(p);return p*p*p*(p*(p*6-15)+10);};
  const SATS=[{url:"https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/BlueMarble_NextGeneration/default/GoogleMapsCompatible_Level8/{z}/{y}/{x}.jpeg",maxz:8,name:"NASA GIBS (Blue Marble)"},
   {url:"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",maxz:9,name:"Esri World Imagery"}];
  let si=0,SAT_URL=SATS[0].url,SAT_MAXZ=SATS[0].maxz,satOk=0,satBad=0,satA=0;
  const sc=document.createElement("canvas"),sg=sc.getContext("2d"),tiles=new Map();   // off-screen layer so the fade-in has no seams
  const setNote=()=>{$("note").textContent="Map: Natural Earth · Satellite imagery: "+SATS[si].name;};
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
  function bestTile(z,x,y){                                   // the tile itself, or a blurrier parent while it loads
   const t=getTile(z,x,y,true);if(t&&t.ok)return{im:t.im,d:0};
   if(z>2)getTile(z-2,x>>2,y>>2,true);
   for(let d=1;d<=z;d++){const a=getTile(z-d,x>>d,y>>d,false);if(a&&a.ok)return{im:a.im,d:d};}
   return null;
  }
  function prefetch(){                                        // world at low detail + the region at full detail, before the user clicks
   for(let z=0;z<=2;z++)for(let x=0;x<(1<<z);x++)for(let y=0;y<(1<<z);y++)getTile(z,x,y,true);
   const px=(bb[3]-bb[2])*.8,py=(bb[1]-bb[0])*.8;
   for(let z=3;z<=SAT_MAXZ;z++){const n=1<<z;
    for(let x=Math.floor(tx(bb[2]-px)*n);x<=Math.floor(tx(bb[3]+px)*n);x++)
     for(let y=Math.max(0,Math.floor(ty(bb[1]+py)*n));y<=Math.min(n-1,Math.floor(ty(bb[0]-py)*n));y++)getTile(z,((x%n)+n)%n,y,true);}
  }
  function tri(im,s0x,s0y,s1x,s1y,s2x,s2y,p0,p1,p2){         // draw image triangle (s0,s1,s2) onto screen triangle (p0,p1,p2)
   const det=(s1x-s0x)*(s2y-s0y)-(s2x-s0x)*(s1y-s0y);if(!det)return;
   const ma=((p1[0]-p0[0])*(s2y-s0y)-(p2[0]-p0[0])*(s1y-s0y))/det,mc=((s1x-s0x)*(p2[0]-p0[0])-(s2x-s0x)*(p1[0]-p0[0]))/det,
         mb=((p1[1]-p0[1])*(s2y-s0y)-(p2[1]-p0[1])*(s1y-s0y))/det,md=((s1x-s0x)*(p2[1]-p0[1])-(s2x-s0x)*(p1[1]-p0[1]))/det,
         me=p0[0]-ma*s0x-mc*s0y,mf=p0[1]-mb*s0x-md*s0y;
   const gx=(p0[0]+p1[0]+p2[0])/3,gy=(p0[1]+p1[1]+p2[1])/3;
   sg.save();sg.beginPath();                                   // clip slightly larger than the triangle so neighbours overlap (no hairline seams)
   [p0,p1,p2].forEach((p,i)=>{const dx=p[0]-gx,dy=p[1]-gy,l=Math.hypot(dx,dy)||1,x=p[0]+dx/l*.7,y=p[1]+dy/l*.7;if(i)sg.lineTo(x,y);else sg.moveTo(x,y);});
   sg.closePath();sg.clip();sg.transform(ma,mb,mc,md,me,mf);sg.drawImage(im,0,0);sg.restore();
  }
  function drawSat(Rr,b){                                     // returns true if any imagery was painted
   let z=Math.min(SAT_MAXZ,Math.max(1,Math.ceil(Math.log2(Rr*TAU*dpr/256))));
   let dmin=1e9,dmax=-1e9,lmin=90,lmax=-90;
   for(let i=0;i<=4;i++)for(let j=0;j<=4;j++){                 // what part of the Earth is in view?
    const p=proj.invert([lerp(b.x0,b.x1,i/4),lerp(b.y0,b.y1,j/4)]);
    if(!p||!isFinite(p[0])||!isFinite(p[1]))return false;
    const dl=((p[0]-cam.lon+540)%360+360)%360-180;
    dmin=Math.min(dmin,dl);dmax=Math.max(dmax,dl);lmin=Math.min(lmin,p[1]);lmax=Math.max(lmax,p[1]);
   }
   if(lmax>84||lmin<-84)return false;
   let n,x0,x1,y0,y1;
   for(;;){n=1<<z;x0=Math.floor(tx(cam.lon+dmin)*n);x1=Math.floor(tx(cam.lon+dmax)*n);
    y0=Math.max(0,Math.floor(ty(lmax)*n));y1=Math.min(n-1,Math.floor(ty(lmin)*n));
    if((x1-x0+1)*(y1-y0+1)<=48||z<=1)break;z--;}               // never ask for more than ~48 tiles at once
   sg.setTransform(dpr,0,0,dpr,0,0);sg.clearRect(0,0,W,H);
   const ts=Rr*TAU/n,sub=ts>360?8:ts>180?4:2,cen=[cam.lon,cam.lat];let drew=false;
   for(let xi=x0;xi<=x1;xi++)for(let yi=y0;yi<=y1;yi++){
    const xt=((xi%n)+n)%n,bt=bestTile(z,xt,yi);if(!bt)continue;
    const P=[];let bad=false,x_min=1e9,x_max=-1e9,y_min=1e9,y_max=-1e9;
    for(let j=0;j<=sub;j++)for(let i=0;i<=sub;i++){           // grid of points over the tile, projected onto the globe
     const lon=(xi+i/sub)/n*360-180,lat=Math.atan(Math.sinh(Math.PI*(1-2*(yi+j/sub)/n)))*57.29577951308232;
     if(d3.geoDistance([lon,lat],cen)>1.52)bad=true;           // behind the horizon
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
   if(!tw){if(mode==="spin")cam.R=R0();if(mode==="focus")cam.R=R1();}dirty=true;}
  new ResizeObserver(resize).observe($("stage"));resize();prefetch();setNote();

  const mk=w=>({land:topojson.feature(w,w.objects.land),b:topojson.mesh(w,w.objects.countries,(a,b)=>a!==b)});
  const B="https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/";
  d3.json(B+"countries-110m.json").then(w=>{L1=mk(w);dirty=true;return d3.json(B+"countries-50m.json");})
   .then(w=>{L2=mk(w);dirty=true;})
   .catch(()=>{if(!L1)$("intro").querySelector("p").textContent="Map data could not load (internet needed).";});

  function drawRegion(){
   const tc=cam.tc,hv=(hover&&mode==="spin")?1:0,pulse=clamp(.8+.2*Math.sin(T/450)+.15*hv,0,1);
   const b=path.bounds(regionGeo);if(!isFinite(b[0][0])||!isFinite(b[1][1]))return;
   const gr=ctx.createLinearGradient(0,b[0][1],0,b[1][1]);
   gr.addColorStop(0,lc([255,212,0,.2],[245,252,255,.8*(1-.88*satA)],tc));gr.addColorStop(1,lc([255,212,0,.2],[80,165,235,.62*(1-.88*satA)],tc));
   ctx.beginPath();path(regionGeo);ctx.fillStyle=gr;ctx.fill();ctx.lineJoin="round";
   ctx.strokeStyle=lc([255,212,0,.22*pulse],[170,230,255,.3],tc);ctx.lineWidth=11;ctx.stroke();
   ctx.shadowColor=lc([255,212,0,.9],[160,225,255,.9],tc);ctx.shadowBlur=16;
   ctx.strokeStyle=lc([255,212,0,pulse],[255,255,255,1],tc);ctx.lineWidth=3;ctx.stroke();ctx.shadowBlur=0;
   ctx.textAlign="center";ctx.shadowColor="rgba(0,0,0,.85)";ctx.shadowBlur=8;
   if(tc>.5){ctx.font="700 15px system-ui,sans-serif";ctx.fillStyle="rgba(255,255,255,"+((tc-.5)*2)+")";
    ctx.fillText(R.name.toUpperCase().split("").join(" "),(b[0][0]+b[1][0])/2,b[0][1]-14);}
   else if(d3.geoDistance([clon,clat],[cam.lon,cam.lat])<1.35){const p=proj([clon,clat]);
    if(p){ctx.font="600 13px system-ui,sans-serif";ctx.fillStyle="rgba(255,255,255,"+(1-tc*2)+")";ctx.fillText(R.name,p[0],p[1]-30);}}
   ctx.shadowBlur=0;
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
   satA=0;                                                     // satellite photo fades in as the camera closes in; the vector map stays underneath as fallback
   const cover=Rd/Math.hypot((x1-x0)/2,(y1-y0)/2),sa=ss((cover-1.02)/1.4);
   if(sa>.01){let ok=false;proj.clipExtent(null);try{ok=drawSat(Rd,{x0:x0,y0:y0,x1:x1,y1:y1});}catch(err){}
    proj.clipExtent([[-60,-60],[W+60,H+60]]);
    if(ok){ctx.globalAlpha=sa;ctx.drawImage(sc,0,0,W,H);ctx.globalAlpha=1;satA=sa;
     if(M){ctx.beginPath();path(M.b);ctx.lineWidth=1;ctx.strokeStyle="rgba(255,255,255,"+(.4*sa)+")";ctx.stroke();}}}
   drawRegion();
   if(zk<1){const rg=ctx.createRadialGradient(cx-.2*Rd,cy-.2*Rd,Rd*.4,cx,cy,Rd);
    rg.addColorStop(0,"rgba(0,0,30,0)");rg.addColorStop(1,"rgba(0,0,30,"+(.5*(1-zk))+")");
    ctx.fillStyle=rg;ctx.beginPath();path({type:"Sphere"});ctx.fill();}
   ctx.restore();
   if(s>0){ctx.beginPath();rr(x0,y0,x1-x0,y1-y0,rad);ctx.strokeStyle="rgba(150,215,255,"+(.4*s)+")";ctx.lineWidth=1.5;ctx.stroke();}
  }

  /* ---- camera tweens: every channel has its own time window and easing ---- */
  function tween(to,dur,win,done,midAt,mid){
   mode="anim";hover=false;dragging=false;uvx=uvy=0;
   tw={from:Object.assign({},cam),to:to,dur:dur,win:win,done:done,t0:performance.now(),midAt:midAt,mid:mid,midDone:!mid,dl:((to.lon-cam.lon+540)%360)-180};
  }
  function stepTween(now){
   const p=clamp((now-tw.t0)/tw.dur),w=tw.win,f=tw.from,t=tw.to;
   const e=k=>ease(clamp((p-w[k][0])/(w[k][1]-w[k][0])));
   cam.lon=f.lon+tw.dl*e("rot");cam.lat=lerp(f.lat,t.lat,e("rot"));
   cam.R=Math.exp(lerp(Math.log(f.R),Math.log(t.R),e("zoom")));
   cam.s=lerp(f.s,t.s,e("slide"));cam.tc=lerp(f.tc,t.tc,e("col"));
   $("intro").style.opacity=t.s>f.s?clamp(1-p*8):clamp((p-.8)*5);
   if(!tw.midDone&&p>=tw.midAt){tw.midDone=true;tw.mid();}
   if(p>=1){cam.lon=t.lon;const d=tw.done;tw=null;d();}
  }
  function enter(){
   tween({lon:clon,lat:clat,R:R1(),s:1,tc:1},4600,{rot:[0,.8],zoom:[.06,.9],slide:[.55,1],col:[.25,.85]},
    ()=>{mode="focus";dirty=true;},.8,()=>{showRegion();panel.classList.add("show");});
  }
  G.leave=function(){
   view="earth";panel.classList.remove("show");
   tween({lon:cam.lon,lat:22,R:R0(),s:0,tc:0},3600,{rot:[.3,1],zoom:[.2,1],slide:[0,.5],col:[.2,.8]},
    ()=>{mode="spin";spinV=0;},0,null);
  };

  /* ---- interaction: drag with inertia, hover pause, click region ---- */
  function hit(x,y){if(mode!=="spin"||tw)return false;
   if(d3.geoDistance([clon,clat],[cam.lon,cam.lat])>1.3)return false;
   const p=proj([clon,clat]);if(!p)return false;
   return Math.hypot(x-p[0],y-p[1])<Math.max(32,cam.R*spanRad/2+14);}
  let lx=0,ly=0,lt=0;
  cvs.addEventListener("pointerdown",e=>{if(mode!=="spin")return;dragging=true;moved=0;lx=e.offsetX;ly=e.offsetY;lt=performance.now();uvx=uvy=0;
   try{cvs.setPointerCapture(e.pointerId);}catch(err){}cvs.style.cursor="grabbing";});
  cvs.addEventListener("pointermove",e=>{
   if(dragging){const now=performance.now(),dts=Math.max(.008,(now-lt)/1000),K=70/cam.R,dx=e.offsetX-lx,dy=e.offsetY-ly;
    cam.lon-=dx*K;cam.lat=clamp(cam.lat+dy*K,-70,70);
    uvx=clamp(.6*uvx+.4*(-dx*K/dts),-150,150);uvy=clamp(.6*uvy+.4*(dy*K/dts),-150,150);
    moved+=Math.abs(dx)+Math.abs(dy);lx=e.offsetX;ly=e.offsetY;lt=now;}
   else if(mode==="spin"){hover=hit(e.offsetX,e.offsetY);cvs.style.cursor=hover?"pointer":"grab";}});
  cvs.addEventListener("pointerup",e=>{if(!dragging)return;dragging=false;cvs.style.cursor="grab";
   if(moved<6&&hit(e.offsetX,e.offsetY))enter();});
  cvs.addEventListener("pointerleave",()=>{hover=false;});

  let last=performance.now();
  function frame(now){
   const dt=Math.min((now-last)/1000,.05);last=now;T=now;
   if(tw)stepTween(now);
   else if(mode==="spin"){
    spinV+=(((hover||dragging)?0:SPEED)-spinV)*(1-Math.exp(-dt*2.2));   // smooth pause / resume
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

 /* =================== PANEL =================== */
 function crumb(parent,text,fn){const a=document.createElement(fn?"a":"span");a.textContent=text;if(fn)a.onclick=fn;parent.appendChild(a);}
 function sep(parent){const i=document.createElement("i");i.textContent="›";parent.appendChild(i);}
 function setNav(){
  const c=$("crumbs");c.innerHTML="";crumb(c,"Earth",()=>G.leave());sep(c);
  crumb(c,"Third Pole Depletion",view==="var"?showRegion:null);
  if(view==="var"){sep(c);crumb(c,cv.label,null);}
  $("nb").textContent=view==="var"?"← All variables":"← Back to Earth";
 }
 function back(){if(view==="var")showRegion();else if(view==="region")G.leave();}
 $("nb").onclick=back;
 document.addEventListener("keydown",e=>{if(e.key==="Escape")back();});

 $("rsub").textContent=R.name+" · satellite and model data, "+Y0+"–"+Y1;
 $("tpdef").textContent="The Third Pole is the Hindu Kush–Himalaya–Tibetan Plateau region. It stores more ice than anywhere else on Earth outside the Arctic and Antarctica, and it feeds rivers that millions of people depend on. “Depletion” means that this ice and snow is being lost. Below, "+V.length+" signals for the "+R.name+" are tested for a trend. The "+R.name+" is famous for glaciers that have held up better than most of the Third Pole.";
 const demos=V.filter(v=>v.demo);
 if(demos.length)$("demo-warn").innerHTML='<div class="warn">⚠ DEMO DATA (synthetic, not real observations) for: '+demos.map(v=>v.label).join(", ")+'. Add the real CSVs to data/'+R.id+'/ to replace it.</div>';

 function drawBars(){
  const rows=V.map(v=>({v:v,s:stat(v,FULL)})).filter(r=>r.s);
  rows.sort((a,b)=>((a.s.p<.05?0:1)-(b.s.p<.05?0:1))||(Math.abs(b.s.tau)-Math.abs(a.s.tau)));
  const nSig=rows.filter(r=>r.s.p<.05).length;
  $("tally").innerHTML="<b>"+nSig+"</b> of "+rows.length+" signals show a statistically significant trend (p &lt; 0.05); <b>"+(rows.length-nSig)+"</b> do not."+
   '<br><span class="dot" style="background:#5ec8ff"></span>significant<span class="dot" style="background:#8794a4"></span>not significant';
  const box=$("bars");box.innerHTML="";
  rows.forEach((r,i)=>{
   const sig=r.s.p<.05,w=Math.min(100,Math.abs(r.s.tau)/.8*100),el=document.createElement("div");
   el.className="row "+(sig?"sig":"non");
   el.innerHTML='<div class="track"><div class="fill" data-w="'+w.toFixed(1)+'" style="transition-delay:'+(i*90)+'ms"></div><div class="name">'+r.v.label+(r.v.demo?"<em>demo</em>":"")+
    '</div></div><div class="val"><b>'+(r.s.tau>0?"↑":"↓")+" τ "+Math.abs(r.s.tau).toFixed(2)+"</b><span>p "+fmtP(r.s.p)+"</span></div>";
   el.onclick=()=>openVar(r.v.key);box.appendChild(el);
  });
  requestAnimationFrame(()=>requestAnimationFrame(()=>box.querySelectorAll(".fill").forEach(f=>{f.style.width=f.dataset.w+"%";})));
 }
 function showRegion(){view="region";$("v-var").style.display="none";$("v-region").style.display="block";panel.scrollTop=0;setNav();drawBars();}

 /* ---- variable page + dual slider ---- */
 const s0=$("y0"),s1=$("y1"),GAP=4;
 [s0,s1].forEach(s=>{s.min=Y0;s.max=Y1;s.step=1;});
 const pos=v=>(v-Y0)/(Y1-Y0),at=f=>"calc(11px + (100% - 22px) * "+f+")";
 for(let y=Y0;y<=Y1;y+=5){const t=document.createElement("div");t.className="tk";t.textContent=y;t.style.left=at(pos(y));$("ticks").appendChild(t);}
 [[Y0,Y1,"Full record"],[2010,Y1,"Since 2010"],[2015,Y1,"Since 2015"],[Y0,2013,Y0+"–2013"]].forEach(p=>{
  if(p[0]<Y0||p[1]>Y1||p[1]-p[0]<GAP)return;const b=document.createElement("button");b.textContent=p[2];
  b.onclick=()=>{s0.value=p[0];s1.value=p[1];sched();};$("presets").appendChild(b);});
 let pend=false;const sched=()=>{if(!pend){pend=true;requestAnimationFrame(()=>{pend=false;drawVar();});}};
 s0.oninput=()=>{if(+s0.value>+s1.value-GAP)s0.value=+s1.value-GAP;sched();};
 s1.oninput=()=>{if(+s1.value<+s0.value+GAP)s1.value=+s0.value+GAP;sched();};

 function openVar(key){
  cv=V.find(v=>v.key===key);if(!cv)return;view="var";
  $("v-region").style.display="none";$("v-var").style.display="block";panel.scrollTop=0;setNav();
  $("vt").textContent=cv.label;$("vsrc").textContent="Data: "+cv.source;
  $("vdef").textContent=cv.definition;$("vwhy").textContent="Why it matters: "+cv.why;
  $("vdemo").innerHTML=cv.demo?'<div class="warn">⚠ DEMO DATA: synthetic, not real. Put the real file at '+cv.file+'</div>':"";
  s0.value=Y0;s1.value=Y1;drawVar();
 }
 function drawVar(){
  const y0=+s0.value,y1=+s1.value,u=cv.unit,BASE={paper_bgcolor:"rgba(0,0,0,0)",plot_bgcolor:"rgba(0,0,0,0)",font:{color:"#eaf6ff",size:12}};
  $("wlabel").textContent=y0+" – "+y1+"  ·  "+(y1-y0+1)+" years";
  $("b0").textContent=y0;$("b1").textContent=y1;$("b0").style.left=at(pos(y0));$("b1").style.left=at(pos(y1));
  $("fill").style.left=at(pos(y0));$("fill").style.width="calc((100% - 22px) * "+(pos(y1)-pos(y0))+")";
  $("presets").querySelectorAll("button").forEach(b=>{b.classList.toggle("on",b.textContent==="Full record"?(y0===Y0&&y1===Y1):false);});
  const s=stat(cv,y0+"-"+y1),idx=[];
  cv.dates.forEach((d,i)=>{const y=+d.slice(0,4);if(y>=y0&&y<=y1)idx.push(i);});
  if(!s||!idx.length){$("verdict").innerHTML="";$("grid").innerHTML="";$("agree").textContent="";
   $("sentence").textContent="Not enough data in this window (at least 5 years with 6+ months each are needed).";Plotly.purge("ts");return;}
  const X=idx.map(i=>cv.dates[i]+"-01"),A=idx.map(i=>cv.anom[i]),W=idx.map(i=>cv.raw[i]),by={};
  idx.forEach(i=>{const y=cv.dates[i].slice(0,4);(by[y]=by[y]||[]).push(cv.anom[i]);});
  const ay=Object.keys(by).filter(y=>by[y].length>=6),am=ay.map(y=>by[y].reduce((a,b)=>a+b,0)/by[y].length);
  const td=d=>+d.slice(0,4)+(+d.slice(5,7)-.5)/12,ends=[X[0],X[X.length-1]],GR={gridcolor:"rgba(255,255,255,.1)"};
  Plotly.react("ts",[
   {x:X,y:W,xaxis:"x",yaxis:"y",mode:"lines",name:"Monthly value",line:{color:"#9fb3c8",width:1}},
   {x:X,y:A,xaxis:"x2",yaxis:"y2",mode:"lines",name:"Monthly anomaly",line:{color:"rgba(94,200,255,.65)",width:1.3}},
   {x:ay.map(y=>y+"-07-01"),y:am,xaxis:"x2",yaxis:"y2",mode:"markers",name:"Yearly mean anomaly",marker:{color:"#fff",size:7}},
   {x:ends,y:ends.map(d=>s.ic+s.slope*td(d)),xaxis:"x2",yaxis:"y2",mode:"lines",name:"Linear trend",line:{color:"#ffb454",width:3}}],
   Object.assign({height:400,margin:{l:62,r:16,t:6,b:70},legend:{orientation:"h",y:-.2},
    xaxis:Object.assign({anchor:"y",matches:"x2",showticklabels:false},GR),
    yaxis:Object.assign({domain:[.72,1],title:{text:"raw ("+u+")",font:{size:11}}},GR),
    xaxis2:Object.assign({anchor:"y2"},GR),
    yaxis2:Object.assign({domain:[0,.64],title:{text:"anomaly ("+u+")",font:{size:11}},zeroline:true,zerolinecolor:"rgba(255,255,255,.5)"},GR)},BASE),
   {displayModeBar:false,responsive:true});
  const sig=s.p<.05,ci0=s.lo>0||s.hi<0,n=y1-y0+1,cell=(k,v)=>"<div>"+k+"<b>"+v+"</b></div>";
  $("verdict").innerHTML='<span class="chip '+(sig?"sig":"non")+'">'+(sig?"Statistically significant":"Not statistically significant")+"</span>";
  $("grid").innerHTML=cell("Rate of change",fmt(s.slope)+" "+u+"/yr")+cell("95% confidence interval",fmt(s.lo)+" to "+fmt(s.hi))+cell("Mann-Kendall p-value",fmtP(s.p))+
   cell("Kendall τ",s.tau.toFixed(2))+cell("Years analysed",s.ny)+cell("Change over window","≈ "+fmt(s.slope*n)+" "+u);
  $("sentence").textContent=cv.label+" in the "+R.name+" "+(s.slope>0?"increased":"decreased")+" by "+fmt(Math.abs(s.slope))+" "+u+" per year between "+y0+" and "+y1+
   " (95% CI "+fmt(s.lo)+" to "+fmt(s.hi)+"; Mann-Kendall p = "+fmtP(s.p)+"), so the trend is "+(sig?"statistically significant":"not statistically significant")+".";
  $("agree").textContent=(sig===ci0)?"Both tests agree.":"The two tests disagree here, so treat this trend with caution.";
 }
 setNav();
}
</script></body></html>
"""

# The globe + panels are one self-contained web component; it is written next to this file automatically.
_dir = Path(__file__).resolve().parent / "globe_component"
_dir.mkdir(exist_ok=True)
(_dir / "index.html").write_text(COMPONENT_HTML, encoding="utf-8")
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


@st.cache_data(show_spinner="Crunching trend statistics for every time window...")
def load(signature):  # signature changes when you add/replace a CSV, so the cache refreshes
    return bk.build_payload()


payload, warnings = load(bk.data_signature())
for w in warnings:
    st.warning(w)

# All statistics (every slider window) are computed in backend.py and handed to the front end,
# so the slider responds instantly with no server round-trip.
_comp(payload=payload, key="main")
