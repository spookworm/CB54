function [KwE] = KopE(wE, input)
KwE = cell(1:2);

for n = 1:2
    KwE{n} = Kop(wE{n}, input.FFTG);
end

dummy = graddiv(KwE, input); % dummy is temporary storage

for n = 1:2
    KwE{n} = KwE{n} - dummy{n} / input.gamma_0^2;
end