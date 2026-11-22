function tree(node, x, strict) {
  while (Array.isArray(node)) {
    const [feature, threshold, left, right] = node;
    const value = Math.fround(x[feature]);
    node = (strict ? value < Math.fround(threshold) : value <= threshold) ? left : right;
  }
  return node;
}
function forest(trees, x) {
  return trees.reduce((sum, t) => sum + tree(t, x, false), 0) / trees.length;
}
export default {
  async fetch(request) {
    const url = new URL(request.url);
    if (url.pathname === '/') return new Response(data.html, {headers:{'Content-Type':'text/html; charset=utf-8'}});
    if (url.pathname === '/api/report') return Response.json(data.report);
    if (url.pathname !== '/api/predict') return new Response('Not found',{status:404});
    const raw = url.searchParams.get('prior');
    const prior = Number(raw);
    if (!raw || !/^\d+$/.test(raw) || prior>350000) return new Response('Invalid arrivals',{status:400});
    const x=[prior,data.report.two_back,data.report.year_back,Math.sin(Math.PI/6),Math.cos(Math.PI/6)];
    const boost=data.boosted.reduce((sum,t)=>Math.fround(sum+Math.fround(tree(t,x,true))),Math.fround(data.base));
    const rf=forest(data.forest,x);
    const stack=data.stack_bias+data.stack_weights[0]*rf+data.stack_weights[1]*boost;
    return Response.json({'Seasonal naive':x[2],'AR(1)':Math.max(0,data.intercept+data.slope*prior),'Random Forest':rf,'Stacking':Math.max(0,stack),'XGBoost':Math.max(0,boost),
      'AR + Forest':Math.max(0,data.intercept+data.slope*prior+forest(data.residual,x))});
  }
};
