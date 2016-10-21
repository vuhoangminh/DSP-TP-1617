function Sinfn_sig = Sinfn(f,Ts,num_sam)
    x=1:num_sam;
    % Define Step signal
    Sinfn_sig=sin(2*pi*f*x*1/Ts);
end
