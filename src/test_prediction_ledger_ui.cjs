// Exercise the shipped filter/render script in a minimal DOM, without opening a browser.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const template=fs.readFileSync(path.join(__dirname,'prediction_ledger.html'),'utf8');
class Element {
  constructor(){this.children=[];this.value='';this.textContent='';this.events={};this.disabled=false;}
  replaceChildren(){this.children=[];this.textContent='';}
  append(child){this.children.push(child);}
  add(option){this.children.push(option);if(this.children.length===1)this.value=option.value;}
  addEventListener(name,callback){this.events[name]=callback;}
}
const base={cohort:'Delivered price alerts',sport:'NBA',book:'fanduel',version:'nba-prop-price-check-v1',market:'points',date:'2026-11-04',event_id:'g1',player:'Test Player',matchup:'PHI @ DET',home_team:'DET',away_team:'PHI',side:'over',line:10.5,price:-110,prediction:null,actual:12,forecast_at:'2026-11-04T23:00:00Z',paper_units:1,source_status:'delivered',reason:'verified',repeated_observations:0};
const rows=[{...base,id:'one',result:'win'},{...base,id:'two',event_id:'g2',player:'Other Player',result:'loss',paper_units:-1,actual:5},
 {...base,id:'void',result:'void',paper_units:0},{...base,id:'later',date:'2026-11-05',result:'pending',paper_units:null,actual:null},
 {...base,id:'dk',book:'draftkings',result:'win'},{...base,id:'new-version',version:'different-version',result:'win'},
 {...base,id:'forecast',cohort:'Stat forecast accuracy',book:'no quoted price',version:'forecast-v1',result:'no_line',side:null,line:null,price:null,prediction:10,actual:12,paper_units:null}];
const nodes={};for(const match of template.matchAll(/id="([^"]+)"/g))nodes[match[1]]=new Element();
for(const id of ['daily','ledger'])nodes[id].tBodies=[new Element()];
nodes.data.textContent=JSON.stringify({records:rows,built_at:'2026-11-05T12:00:00Z',since:'2026-09-01',through:'2026-11-05',notes:[],source_health:{}});
const context=vm.createContext({document:{getElementById:id=>nodes[id],createElement:()=>new Element()},
 Option:function(text,value){return {textContent:text,value};},console,setTimeout,URL,Blob});
vm.runInContext(template.match(/<script>\s*([\s\S]*?)<\/script>/)[1],context);
assert.equal(Number(nodes.wins.textContent),1);assert.equal(Number(nodes.losses.textContent),1);
assert.equal(nodes.rate.textContent,'50.0%');assert.equal(nodes.ledger.tBodies[0].children.length,4);
nodes.search.value='Other Player';nodes.search.events.input();
assert.equal(Number(nodes.wins.textContent),0);assert.equal(Number(nodes.losses.textContent),1);
assert.equal(nodes.ledger.tBodies[0].children.length,1);
nodes.search.value='';nodes.search.events.input();nodes.through.value='2026-11-04';nodes.through.events.input();
assert.equal(Number(nodes.pending.textContent),0);assert.equal(nodes.ledger.tBodies[0].children.length,3);
nodes.cohort.value='Stat forecast accuracy';nodes.cohort.events.change();
assert.equal(nodes.ledger.tBodies[0].children.length,1);assert.equal(nodes.rate.textContent,'—');
assert.match(nodes.mae.textContent,/2$/);assert.equal(nodes.other.textContent,'1 / 0');
nodes.search.value='no match';nodes.search.events.input();assert.equal(nodes.empty.hidden,false);
assert.equal(nodes.prev.disabled,true);assert.equal(nodes.next.disabled,true);
console.log('Ledger UI passed: strategy/book/version isolation, search, dates, no-line accuracy, empty state.');
