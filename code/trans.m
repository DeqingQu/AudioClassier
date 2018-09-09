clear

[y,fs] = audioread('../dataset/s1/bbaf2n.wav'); 
% read the wav file
fs2 = fs/10;
% change the sample rate to 5000Hz
filename = 'handle.wav';
% build a new wav file
audiowrite(filename,y,fs2);
% write a new wav file
clear y fs2

[y,fs2] = audioread(filename);
t = linspace(0,length(y)/fs2,length(y));
plot(t,y)