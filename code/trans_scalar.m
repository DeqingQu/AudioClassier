clc
close all

[y,fs] = audioread('bbal3a_3.wav');
window = hamming(64);
noverlap = 48;
nfft = 1024;
[S,F,T,P] = spectrogram(y,window,noverlap,nfft,fs,'yaxis');


surf(T,F,10*log10(P),'edgecolor','none');
% surf(T,F,P,'edgecolor','none');
axis tight;
view(0,90);
% s = spectrogram(y,window,noverlap);
% plot(s)
colormap(jet);
set(gca,'clim',[-80,-30]);
xlabel('Temps(seconds)');
ylabel('Frequences,Hz');