function image_object_render(image_object, map, materials, markerColor, title_str)
image_object_render = ind2rgb(image_object, map);
figure
imagesc(image_object_render, 'XData', 1/2, 'YData', 1/2)
title(title_str)
legend
set(gcf, 'units', 'normalized', 'outerposition', [0, 0, 1, 1])
hold on
L = plot(ones(height(materials)), 'LineStyle', 'none', 'marker', 's', 'visible', 'on');
set(L, {'MarkerFaceColor'}, markerColor, {'MarkerEdgeColor'}, markerColor);
colormap(map)
legend(materials)
end