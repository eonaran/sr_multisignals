function a = cummax_flip(a)
a=a(end:-1:1);
t = a(1);
for k = 2:length(a)
  if a(k) < t
    a(k) = t;
  else
    t = a(k);
  end
end
a=a(end:-1:1);