    /* -- load_cache_elem.mako -- */
    CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)descr->data;
%if locality_stats:
<%namespace file="cache_stats_macro.mako" import="*"/>
    CMLQCacheStatsElem *cache_stats = &elem->stats;
    cache_stats->opname = "${opname}";
<%count_stat("op_exec_count")%>
%endif
