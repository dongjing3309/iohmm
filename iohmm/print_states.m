function print_states(states)
%PRINT_STATES Compactly print states or measurements in a line

sstr = [];
for i=1:numel(states)
    sstr = strcat(sstr, num2str(states(i)));
end
fprintf('%s\n', sstr)

end

