% Nolte, David D
% github.itap.purdue.edu/nolte

% function [A,degree,Lap] = Erdos(N,p)
% Generates an Erdos-Renyi random graph of N nodes with edge probability p
% A is the adjacency matrix
% degree is the degree of the node
% Lap is the Lapacian matrix



function [A,degree,Lap] = Erdos(N,p)

e = round(p*N*(N-1)/2);

A = zeros(N,N);         %Adjacency matrix

loop = 0;
while loop ~=e

%     x = round(rand*(N-1))+1;
%     y = round(rand*(N-1))+1;
    x = ceil(N*rand);
    y = ceil(N*rand);

    flag = A(y,x);
    if (x~=y)&(flag==0)
        A(x,y) = 1;
        A(y,x) = 1;
        loop = loop +1;
    end
    

end

degree = sum(A);

Lap = -A;
for loop = 1:N
    Lap(loop,loop) = degree(loop);
end


