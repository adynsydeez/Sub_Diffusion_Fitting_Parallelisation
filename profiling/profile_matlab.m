% Profiles the original MATLAB fit on one slice with the MATLAB Profiler.
% Run:  matlab -batch "run('profiling/profile_matlab.m')"
% Writes the interactive HTML report to profiling/matlab/html/ and the top functions by
% self time to profiling/matlab/top_functions.txt.

root = fileparts(fileparts(mfilename('fullpath'))); % run() changes into this script's folder
cd(root);
addpath(fullfile(root, 'MATLAB'));
slice = 73;
out = fullfile(root, 'profiling', 'matlab');

profile clear
profile on
run_fit(slice, fullfile(out, 'fit_outputs'));
profile off

p = profile('info');
profsave(p, fullfile(out, 'html'));

% Self time = time in the function itself, excluding the functions it calls
ft = p.FunctionTable;
self = arrayfun(@(f) f.TotalTime - sum([f.Children.TotalTime]), ft);
total = max([ft.TotalTime]); % run_fit, which contains everything else
[~, order] = sort(self, 'descend');

fid = fopen(fullfile(out, 'top_functions.txt'), 'w');
for dst = [1 fid]
    fprintf(dst, 'MATLAB Profiler, run_fit(%d): %.1f s total\n\n', slice, total);
    fprintf(dst, '%-58s %9s %9s %9s %8s\n', 'Function', 'Calls', 'Total s', 'Self s', 'Self %');
    for i = order(1:min(20, numel(order)))'
        name = ft(i).FunctionName;
        if numel(name) > 58, name = ['...' name(end-54:end)]; end
        fprintf(dst, '%-58s %9d %9.2f %9.2f %7.1f%%\n', name, ft(i).NumCalls, ft(i).TotalTime, self(i), ...
                100 * self(i) / total);
    end
end
fclose(fid);
