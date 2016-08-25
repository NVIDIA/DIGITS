local daa='1.4.0'local _ba,aba,bba,cba,dba=next,type,unpack,select,pcall
local _ca,aca=setmetatable,getmetatable;local bca,cca=table.insert,table.sort;local dca,_da=table.remove,table.concat
local ada,bda,cda=math.randomseed,math.random,math.huge;local dda,__b,a_b=math.floor,math.max,math.min;local b_b=rawget;local c_b=bba
local d_b,_ab=pairs,ipairs;local aab={}local function bab(bcb,ccb)return bcb>ccb end
local function cab(bcb,ccb)return bcb<ccb end;local function dab(bcb,ccb,dcb)
return(bcb<ccb)and ccb or(bcb>dcb and dcb or bcb)end
local function _bb(bcb,ccb)return ccb and true end;local function abb(bcb)return not bcb end;local function bbb(bcb)local ccb=0
for dcb,_db in d_b(bcb)do ccb=ccb+1 end;return ccb end
local function cbb(bcb,ccb,dcb,...)local _db
local adb=dcb or aab.identity;for bdb,cdb in d_b(bcb)do
if not _db then _db=adb(cdb,...)else local ddb=adb(cdb,...)_db=
ccb(_db,ddb)and _db or ddb end end;return _db end
local function dbb(bcb,ccb,dcb)for i=0,#bcb,ccb do local _db=aab.slice(bcb,i+1,i+ccb)
if#_db>0 then dcb(_db)end end end
local function _cb(bcb,ccb,dcb)if ccb==0 then dcb(bcb)end
for i=1,ccb do bcb[ccb],bcb[i]=bcb[i],bcb[ccb]_cb(bcb,ccb-
1,dcb)bcb[ccb],bcb[i]=bcb[i],bcb[ccb]end end;local acb=-1
function aab.each(bcb,ccb,...)for dcb,_db in d_b(bcb)do ccb(dcb,_db,...)end end
function aab.eachi(bcb,ccb,...)
local dcb=aab.sort(aab.select(aab.keys(bcb),function(_db,adb)return aab.isInteger(adb)end))for _db,adb in _ab(dcb)do ccb(adb,bcb[adb],...)end end
function aab.at(bcb,...)local ccb={}for dcb,_db in _ab({...})do
if aab.has(bcb,_db)then ccb[#ccb+1]=bcb[_db]end end;return ccb end
function aab.count(bcb,ccb)if aab.isNil(ccb)then return aab.size(bcb)end;local dcb=0
aab.each(bcb,function(_db,adb)if
aab.isEqual(adb,ccb)then dcb=dcb+1 end end)return dcb end
function aab.countf(bcb,ccb,...)return aab.count(aab.map(bcb,ccb,...),true)end
function aab.cycle(bcb,ccb)ccb=ccb or 1;if ccb<=0 then return function()end end;local dcb,_db;local adb=0
while true do
return
function()dcb=
dcb and _ba(bcb,dcb)or _ba(bcb)_db=
not _db and dcb or _db;if ccb then
adb=(dcb==_db)and adb+1 or adb;if adb>ccb then return end end
return dcb,bcb[dcb]end end end;function aab.map(bcb,ccb,...)local dcb={}
for _db,adb in d_b(bcb)do dcb[_db]=ccb(_db,adb,...)end;return dcb end;function aab.reduce(bcb,ccb,dcb)
for _db,adb in
d_b(bcb)do if dcb==nil then dcb=adb else dcb=ccb(dcb,adb)end end;return dcb end;function aab.reduceRight(bcb,ccb,dcb)return
aab.reduce(aab.reverse(bcb),ccb,dcb)end
function aab.mapReduce(bcb,ccb,dcb)
local _db={}for adb,bdb in d_b(bcb)do _db[adb]=not dcb and bdb or ccb(dcb,bdb)
dcb=_db[adb]end;return _db end;function aab.mapReduceRight(bcb,ccb,dcb)
return aab.mapReduce(aab.reverse(bcb),ccb,dcb)end
function aab.include(bcb,ccb)local dcb=
aab.isFunction(ccb)and ccb or aab.isEqual;for _db,adb in d_b(bcb)do if dcb(adb,ccb)then
return true end end;return false end
function aab.detect(bcb,ccb)
local dcb=aab.isFunction(ccb)and ccb or aab.isEqual;for _db,adb in d_b(bcb)do if dcb(adb,ccb)then return _db end end end
function aab.contains(bcb,ccb)return aab.toBoolean(aab.detect(bcb,ccb))end
function aab.findWhere(bcb,ccb)
local dcb=aab.detect(bcb,function(_db)for adb in d_b(ccb)do
if ccb[adb]~=_db[adb]then return false end end;return true end)return dcb and bcb[dcb]end
function aab.select(bcb,ccb,...)local dcb=aab.map(bcb,ccb,...)local _db={}for adb,bdb in d_b(dcb)do if bdb then
_db[#_db+1]=bcb[adb]end end;return _db end
function aab.reject(bcb,ccb,...)local dcb=aab.map(bcb,ccb,...)local _db={}for adb,bdb in d_b(dcb)do if not bdb then
_db[#_db+1]=bcb[adb]end end;return _db end;function aab.all(bcb,ccb,...)return
( (#aab.select(aab.map(bcb,ccb,...),_bb))== (#bcb))end
function aab.invoke(bcb,ccb,...)
local dcb={...}
return
aab.map(bcb,function(_db,adb)
if aab.isTable(adb)then
if aab.has(adb,ccb)then if aab.isCallable(adb[ccb])then return
adb[ccb](adb,c_b(dcb))else return adb[ccb]end else if
aab.isCallable(ccb)then return ccb(adb,c_b(dcb))end end elseif aab.isCallable(ccb)then return ccb(adb,c_b(dcb))end end)end
function aab.pluck(bcb,ccb)return
aab.reject(aab.map(bcb,function(dcb,_db)return _db[ccb]end),abb)end;function aab.max(bcb,ccb,...)return cbb(bcb,bab,ccb,...)end;function aab.min(bcb,ccb,...)return
cbb(bcb,cab,ccb,...)end
function aab.shuffle(bcb,ccb)if ccb then ada(ccb)end
local dcb={}
aab.each(bcb,function(_db,adb)local bdb=dda(bda()*_db)+1;dcb[_db]=dcb[bdb]
dcb[bdb]=adb end)return dcb end
function aab.same(bcb,ccb)
return
aab.all(bcb,function(dcb,_db)return aab.include(ccb,_db)end)and
aab.all(ccb,function(dcb,_db)return aab.include(bcb,_db)end)end;function aab.sort(bcb,ccb)cca(bcb,ccb)return bcb end
function aab.groupBy(bcb,ccb,...)local dcb={...}
local _db={}
local adb=aab.isFunction(ccb)and ccb or(aab.isString(ccb)and function(bdb,cdb)return
cdb[ccb](cdb,c_b(dcb))end)if not adb then return end
aab.each(bcb,function(bdb,cdb)local ddb=adb(bdb,cdb)
if _db[ddb]then _db[ddb][#_db[ddb]+
1]=cdb else _db[ddb]={cdb}end end)return _db end
function aab.countBy(bcb,ccb,...)local dcb={...}local _db={}
aab.each(bcb,function(adb,bdb)local cdb=ccb(adb,bdb,c_b(dcb))_db[cdb]=(
_db[cdb]or 0)+1 end)return _db end
function aab.size(...)local bcb={...}local ccb=bcb[1]if aab.isNil(ccb)then return 0 elseif aab.isTable(ccb)then return
bbb(bcb[1])else return bbb(bcb)end end;function aab.containsKeys(bcb,ccb)
for dcb in d_b(ccb)do if not bcb[dcb]then return false end end;return true end
function aab.sameKeys(bcb,ccb)
aab.each(bcb,function(dcb)if
not ccb[dcb]then return false end end)
aab.each(ccb,function(dcb)if not bcb[dcb]then return false end end)return true end;function aab.toArray(...)return{...}end
function aab.find(bcb,ccb,dcb)for i=dcb or 1,#bcb do if
aab.isEqual(bcb[i],ccb)then return i end end end
function aab.reverse(bcb)local ccb={}for i=#bcb,1,-1 do ccb[#ccb+1]=bcb[i]end;return ccb end
function aab.selectWhile(bcb,ccb,...)local dcb={}for _db,adb in _ab(bcb)do
if ccb(_db,adb,...)then dcb[_db]=adb else break end end;return dcb end
function aab.dropWhile(bcb,ccb,...)local dcb
for _db,adb in _ab(bcb)do if not ccb(_db,adb,...)then dcb=_db;break end end;if aab.isNil(dcb)then return{}end;return aab.rest(bcb,dcb)end
function aab.sortedIndex(bcb,ccb,dcb,_db)local adb=dcb or cab;if _db then aab.sort(bcb,adb)end;for i=1,#bcb do if not
adb(bcb[i],ccb)then return i end end
return#bcb+1 end
function aab.indexOf(bcb,ccb)for k=1,#bcb do if bcb[k]==ccb then return k end end end
function aab.lastIndexOf(bcb,ccb)local dcb=aab.indexOf(aab.reverse(bcb),ccb)if dcb then return
#bcb-dcb+1 end end;function aab.addTop(bcb,...)
aab.each({...},function(ccb,dcb)bca(bcb,1,dcb)end)return bcb end;function aab.push(bcb,...)aab.each({...},function(ccb,dcb)
bcb[#bcb+1]=dcb end)
return bcb end
function aab.pop(bcb,ccb)
ccb=a_b(ccb or 1,#bcb)local dcb={}
for i=1,ccb do local _db=bcb[1]dcb[#dcb+1]=_db;dca(bcb,1)end;return c_b(dcb)end
function aab.unshift(bcb,ccb)ccb=a_b(ccb or 1,#bcb)local dcb={}for i=1,ccb do local _db=bcb[#bcb]
dcb[#dcb+1]=_db;dca(bcb)end;return c_b(dcb)end
function aab.pull(bcb,...)
for ccb,dcb in _ab({...})do for i=#bcb,1,-1 do
if aab.isEqual(bcb[i],dcb)then dca(bcb,i)end end end;return bcb end
function aab.removeRange(bcb,ccb,dcb)local _db=aab.clone(bcb)local adb,bdb=(_ba(_db)),#_db
if bdb<1 then return _db end;ccb=dab(ccb or adb,adb,bdb)
dcb=dab(dcb or bdb,adb,bdb)if dcb<ccb then return _db end;local cdb=dcb-ccb+1;local ddb=ccb;while cdb>0 do
dca(_db,ddb)cdb=cdb-1 end;return _db end
function aab.chunk(bcb,ccb,...)if not aab.isArray(bcb)then return bcb end;local dcb,_db,adb={},0
local bdb=aab.map(bcb,ccb,...)
aab.each(bdb,function(cdb,ddb)adb=(adb==nil)and ddb or adb;_db=(
(ddb~=adb)and(_db+1)or _db)
if not dcb[_db]then dcb[_db]={bcb[cdb]}else dcb[_db][
#dcb[_db]+1]=bcb[cdb]end;adb=ddb end)return dcb end
function aab.slice(bcb,ccb,dcb)return
aab.select(bcb,function(_db)return
(_db>= (ccb or _ba(bcb))and _db<= (dcb or#bcb))end)end;function aab.first(bcb,ccb)local dcb=ccb or 1
return aab.slice(bcb,1,a_b(dcb,#bcb))end
function aab.initial(bcb,ccb)
if ccb and ccb<0 then return end;return
aab.slice(bcb,1,ccb and#bcb- (a_b(ccb,#bcb))or#bcb-1)end;function aab.last(bcb,ccb)if ccb and ccb<=0 then return end
return aab.slice(bcb,ccb and
#bcb-a_b(ccb-1,#bcb-1)or 2,#bcb)end;function aab.rest(bcb,ccb)if ccb and
ccb>#bcb then return{}end
return aab.slice(bcb,
ccb and __b(1,a_b(ccb,#bcb))or 1,#bcb)end;function aab.compact(bcb)
return aab.reject(bcb,function(ccb,dcb)return
not dcb end)end
function aab.flatten(bcb,ccb)local dcb=ccb or false
local _db;local adb={}
for bdb,cdb in d_b(bcb)do
if aab.isTable(cdb)then
_db=dcb and cdb or aab.flatten(cdb)
aab.each(_db,function(ddb,__c)adb[#adb+1]=__c end)else adb[#adb+1]=cdb end end;return adb end
function aab.difference(bcb,ccb)if not ccb then return aab.clone(bcb)end;return
aab.select(bcb,function(dcb,_db)return not
aab.include(ccb,_db)end)end
function aab.union(...)return aab.uniq(aab.flatten({...}))end
function aab.intersection(bcb,...)local ccb={...}local dcb={}
for _db,adb in _ab(bcb)do if
aab.all(ccb,function(bdb,cdb)return aab.include(cdb,adb)end)then bca(dcb,adb)end end;return dcb end
function aab.symmetricDifference(bcb,ccb)return
aab.difference(aab.union(bcb,ccb),aab.intersection(bcb,ccb))end
function aab.unique(bcb)local ccb={}for i=1,#bcb do if not aab.find(ccb,bcb[i])then
ccb[#ccb+1]=bcb[i]end end;return ccb end
function aab.isunique(bcb)return aab.isEqual(bcb,aab.unique(bcb))end
function aab.zip(...)local bcb={...}
local ccb=aab.max(aab.map(bcb,function(_db,adb)return#adb end))local dcb={}for i=1,ccb do dcb[i]=aab.pluck(bcb,i)end;return dcb end
function aab.append(bcb,ccb)local dcb={}for _db,adb in _ab(bcb)do dcb[_db]=adb end;for _db,adb in _ab(ccb)do
dcb[#dcb+1]=adb end;return dcb end
function aab.interleave(...)return aab.flatten(aab.zip(...))end;function aab.interpose(bcb,ccb)return
aab.flatten(aab.zip(ccb,aab.rep(bcb,#ccb-1)))end
function aab.range(...)
local bcb={...}local ccb,dcb,_db
if#bcb==0 then return{}elseif#bcb==1 then dcb,ccb,_db=bcb[1],0,1 elseif#bcb==2 then
ccb,dcb,_db=bcb[1],bcb[2],1 elseif#bcb==3 then ccb,dcb,_db=bcb[1],bcb[2],bcb[3]end;if(_db and _db==0)then return{}end;local adb={}
local bdb=__b(dda((dcb-ccb)/_db),0)for i=1,bdb do adb[#adb+1]=ccb+_db*i end;if#adb>0 then
bca(adb,1,ccb)end;return adb end
function aab.rep(bcb,ccb)local dcb={}for i=1,ccb do dcb[#dcb+1]=bcb end;return dcb end
function aab.partition(bcb,ccb)return
coroutine.wrap(function()dbb(bcb,ccb or 1,coroutine.yield)end)end
function aab.permutation(bcb)return
coroutine.wrap(function()_cb(bcb,#bcb,coroutine.yield)end)end;function aab.invert(bcb)local ccb={}
aab.each(bcb,function(dcb,_db)ccb[_db]=dcb end)return ccb end
function aab.concat(bcb,ccb,dcb,_db)
local adb=aab.map(bcb,function(bdb,cdb)return
tostring(cdb)end)return _da(adb,ccb,dcb or 1,_db or#bcb)end;function aab.identity(bcb)return bcb end
function aab.once(bcb)local ccb=0;local dcb={}return
function(...)ccb=ccb+1;if ccb<=1 then
dcb={...}end;return bcb(c_b(dcb))end end
function aab.memoize(bcb,ccb)local dcb=_ca({},{__mode='kv'})
local _db=ccb or aab.identity
return function(...)local adb=_db(...)local bdb=dcb[adb]
if not bdb then dcb[adb]=bcb(...)end;return dcb[adb]end end
function aab.after(bcb,ccb)local dcb,_db=ccb,0;return
function(...)_db=_db+1;if _db>=dcb then return bcb(...)end end end
function aab.compose(...)local bcb=aab.reverse{...}return
function(...)local ccb;for dcb,_db in _ab(bcb)do ccb=ccb and _db(ccb)or
_db(...)end;return ccb end end
function aab.pipe(bcb,...)return aab.compose(...)(bcb)end
function aab.complement(bcb)return function(...)return not bcb(...)end end;function aab.juxtapose(bcb,...)local ccb={}
aab.each({...},function(dcb,_db)ccb[#ccb+1]=_db(bcb)end)return c_b(ccb)end
function aab.wrap(bcb,ccb)return function(...)return
ccb(bcb,...)end end
function aab.times(bcb,ccb,...)local dcb={}for i=1,bcb do dcb[i]=ccb(i,...)end;return dcb end
function aab.bind(bcb,ccb)return function(...)return bcb(ccb,...)end end
function aab.bindn(bcb,...)local ccb={...}return function(...)
return bcb(c_b(aab.append(ccb,{...})))end end
function aab.uniqueId(bcb,...)acb=acb+1
if bcb then if aab.isString(bcb)then return bcb:format(acb)elseif
aab.isFunction(bcb)then return bcb(acb,...)end end;return acb end;function aab.keys(bcb)local ccb={}
aab.each(bcb,function(dcb)ccb[#ccb+1]=dcb end)return ccb end;function aab.values(bcb)local ccb={}
aab.each(bcb,function(dcb,_db)ccb[
#ccb+1]=_db end)return ccb end;function aab.toBoolean(bcb)return
not not bcb end
function aab.extend(bcb,...)local ccb={...}
aab.each(ccb,function(dcb,_db)
if aab.isTable(_db)then aab.each(_db,function(adb,bdb)
bcb[adb]=bdb end)end end)return bcb end
function aab.functions(bcb,ccb)bcb=bcb or aab;local dcb={}
aab.each(bcb,function(adb,bdb)if aab.isFunction(bdb)then
dcb[#dcb+1]=adb end end)if not ccb then return aab.sort(dcb)end;local _db=aca(bcb)
if
_db and _db.__index then local adb=aab.functions(_db.__index)aab.each(adb,function(bdb,cdb)
dcb[#dcb+1]=cdb end)end;return aab.sort(dcb)end
function aab.clone(bcb,ccb)if not aab.isTable(bcb)then return bcb end;local dcb={}
aab.each(bcb,function(_db,adb)if
aab.isTable(adb)then
if not ccb then dcb[_db]=aab.clone(adb,ccb)else dcb[_db]=adb end else dcb[_db]=adb end end)return dcb end;function aab.tap(bcb,ccb,...)ccb(bcb,...)return bcb end;function aab.has(bcb,ccb)return
bcb[ccb]~=nil end
function aab.pick(bcb,...)local ccb=aab.flatten{...}
local dcb={}
aab.each(ccb,function(_db,adb)
if not aab.isNil(bcb[adb])then dcb[adb]=bcb[adb]end end)return dcb end
function aab.omit(bcb,...)local ccb=aab.flatten{...}local dcb={}
aab.each(bcb,function(_db,adb)if
not aab.include(ccb,_db)then dcb[_db]=adb end end)return dcb end;function aab.template(bcb,ccb)
aab.each(ccb or{},function(dcb,_db)if not bcb[dcb]then bcb[dcb]=_db end end)return bcb end
function aab.isEqual(bcb,ccb,dcb)
local _db=aba(bcb)local adb=aba(ccb)if _db~=adb then return false end
if _db~='table'then return(bcb==ccb)end;local bdb=aca(bcb)local cdb=aca(ccb)
if dcb then if
bdb or cdb and bdb.__eq or cdb.__eq then return(bcb==ccb)end end
if aab.size(bcb)~=aab.size(ccb)then return false end
for ddb,__c in d_b(bcb)do local a_c=ccb[ddb]if
aab.isNil(a_c)or not aab.isEqual(__c,a_c,dcb)then return false end end
for ddb,__c in d_b(ccb)do local a_c=bcb[ddb]if aab.isNil(a_c)then return false end end;return true end
function aab.result(bcb,ccb,...)
if bcb[ccb]then if aab.isCallable(bcb[ccb])then return bcb[ccb](bcb,...)else return
bcb[ccb]end end;if aab.isCallable(ccb)then return ccb(bcb,...)end end;function aab.isTable(bcb)return aba(bcb)=='table'end
function aab.isCallable(bcb)return
(
aab.isFunction(bcb)or
(aab.isTable(bcb)and aca(bcb)and aca(bcb).__call~=nil)or false)end
function aab.isArray(bcb)if not aab.isTable(bcb)then return false end;local ccb=0
for dcb in
d_b(bcb)do ccb=ccb+1;if aab.isNil(bcb[ccb])then return false end end;return true end
function aab.isIterable(bcb)return aab.toBoolean((dba(d_b,bcb)))end
function aab.isEmpty(bcb)if aab.isNil(bcb)then return true end;if aab.isString(bcb)then
return#bcb==0 end
if aab.isTable(bcb)then return _ba(bcb)==nil end;return true end;function aab.isString(bcb)return aba(bcb)=='string'end;function aab.isFunction(bcb)return
aba(bcb)=='function'end;function aab.isNil(bcb)
return bcb==nil end
function aab.isNumber(bcb)return aba(bcb)=='number'end
function aab.isNaN(bcb)return aab.isNumber(bcb)and bcb~=bcb end
function aab.isFinite(bcb)if not aab.isNumber(bcb)then return false end;return
bcb>-cda and bcb<cda end;function aab.isBoolean(bcb)return aba(bcb)=='boolean'end
function aab.isInteger(bcb)return
aab.isNumber(bcb)and dda(bcb)==bcb end
do aab.forEach=aab.each;aab.forEachi=aab.eachi;aab.loop=aab.cycle
aab.collect=aab.map;aab.inject=aab.reduce;aab.foldl=aab.reduce
aab.injectr=aab.reduceRight;aab.foldr=aab.reduceRight;aab.mapr=aab.mapReduce
aab.maprr=aab.mapReduceRight;aab.any=aab.include;aab.some=aab.include;aab.filter=aab.select
aab.discard=aab.reject;aab.every=aab.all;aab.takeWhile=aab.selectWhile
aab.rejectWhile=aab.dropWhile;aab.shift=aab.pop;aab.remove=aab.pull;aab.rmRange=aab.removeRange
aab.chop=aab.removeRange;aab.sub=aab.slice;aab.head=aab.first;aab.take=aab.first
aab.tail=aab.rest;aab.skip=aab.last;aab.without=aab.difference;aab.diff=aab.difference
aab.symdiff=aab.symmetricDifference;aab.xor=aab.symmetricDifference;aab.uniq=aab.unique
aab.isuniq=aab.isunique;aab.part=aab.partition;aab.perm=aab.permutation;aab.mirror=aab.invert
aab.join=aab.concat;aab.cache=aab.memoize;aab.juxt=aab.juxtapose;aab.uid=aab.uniqueId
aab.methods=aab.functions;aab.choose=aab.pick;aab.drop=aab.omit;aab.defaults=aab.template
aab.compare=aab.isEqual end
do local bcb={}local ccb={}ccb.__index=bcb;local function dcb(_db)local adb={_value=_db,_wrapped=true}
return _ca(adb,ccb)end
_ca(ccb,{__call=function(_db,adb)return dcb(adb)end,__index=function(_db,adb,...)return
bcb[adb]end})function ccb.chain(_db)return dcb(_db)end
function ccb:value()return self._value end;bcb.chain,bcb.value=ccb.chain,ccb.value
for _db,adb in d_b(aab)do
bcb[_db]=function(bdb,...)local cdb=aab.isTable(bdb)and
bdb._wrapped or false
if cdb then
local ddb=bdb._value;local __c=adb(ddb,...)return dcb(__c)else return adb(bdb,...)end end end
bcb.import=function(_db,adb)_db=_db or _G;local bdb=aab.functions()
aab.each(bdb,function(cdb,ddb)
if b_b(_db,ddb)then if not adb then
_db[ddb]=aab[ddb]end else _db[ddb]=aab[ddb]end end)return _db end;ccb._VERSION='Moses v'..daa
ccb._URL='http://github.com/Yonaba/Moses'
ccb._LICENSE='MIT <http://raw.githubusercontent.com/Yonaba/Moses/master/LICENSE>'ccb._DESCRIPTION='utility-belt library for functional programming in Lua'return
ccb end
