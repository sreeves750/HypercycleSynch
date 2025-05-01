% Nolte, David D
% github.itap.purdue.edu/nolte

% function node = makeER(N,p)
% Creates an Erdos-Renyi graph of N nodes and link probability p
% node structure is ...
% node(1).element = node number;
% node(1).numlink = number_of_links;
% node(1).link = [set of linked node numbers];


function node = makeER(N,p)

% Set node structure
node(1).element = 1;
node(1).numlink = 0;
node(1).link = [];

% Create ER graph
[A,degree,Lap] = Erdos(N,p);

% set links
for rowloop = 2:N

    node = addnode(rowloop,node);
    
    for coloop = 1:rowloop-1
        if A(rowloop, coloop) ==1
            node = addlink(rowloop,coloop,node);
        end
    
    
    
    end     % end coloop
end     % end rowloop