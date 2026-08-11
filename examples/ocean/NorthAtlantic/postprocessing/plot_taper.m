
FS=16;
LW=3;

a=2000; b=1000; c=400;
x=linspace(-0.01,0.01,1000);

foo = @(a, x) max(0,tanh((a*x).^3));

plot(x,foo(a,x),'-',x,foo(b,x),'--',x,foo(c,x),':','LineWidth',LW);
set(gca,'FontSize',FS);
