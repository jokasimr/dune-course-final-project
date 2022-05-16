import time
import json
import uuid

from dune.common import comm


class RunInformationCollector:
    def __init__(self, name, vectors, **information):
        self.vectors = vectors
        self.start_time = time.time()
        if comm.rank == 0:
            self.data = dict(name=name, id=str(uuid.uuid4())[:8], **information)
            self.events = []
        
    def adaptivity_event(self):
        vector_sizes = {name: dict(size=comm.sum(v.size)) for name, v in self.vectors.items()}
        if comm.rank == 0:
            self.events.append(('adaptivity', dict(data=vector_sizes, **self.runtime())))
            
    def step_event(self, t):
        if comm.rank == 0:
            self.events.append(('step', dict(**self.runtime(), t=t)))
            
    def solve_event(self, kind):
        start = time.time()
        def done(info):
            end = time.time()
            if comm.rank == 0:
                self.events.append(('solve', dict(kind=kind, **self.runtime(), solve_time=end-start, **info)))
        return done
    
    def runtime(self):
        t = time.time() - self.start_time
        return dict(time=t, time_min=round(t/60, 2))
    
    def save(self):
        if comm.rank == 0:
            with open(self.filename, 'w') as f:
                json.dump(dict(**self.data, events=self.events), f)
        
    @property
    def filename(self):
        return f'{self.data["name"]}_{self.data["id"]}.json'
            
        