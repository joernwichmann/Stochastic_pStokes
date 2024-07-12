from typing import Protocol

class ProcessObject(Protocol):
    def update(self,*args,**kwargs) -> None:
        print("update is not implemented.")
        return
    
    def save(self,*args,**kwargs) -> None:
        print("save is not implemented.")
        return

    def plot(self,*args,**kwargs) -> None:
        print("plot is not implemented.")
        return
    
    def plot_individual(self,*args,**kwargs) -> None:
        print("plot individual is not implemented.")
        return

    def __str__(self) -> str:
        return "__str__ is not implemented."

class ProcessManager:
    """Class that bundles routines for process objects."""
    def __init__(self, list_of_process_objects: list[ProcessObject] = []) -> None:
        self.list_of_process_objects = list_of_process_objects

    def add_process_object(self, process_object: ProcessObject) -> None:
        self.list_of_process_objects.append(process_object)

    def update(self,*args,**kwargs) -> None:
        for process_object in self.list_of_process_objects:
            process_object.update(*args,**kwargs)

    def save(self,*args,**kwargs) -> None:
        for process_object in self.list_of_process_objects:
            process_object.save(*args,**kwargs)
    
    def plot(self,*args,**kwargs) -> None:
        for process_object in self.list_of_process_objects:
            process_object.plot(*args,**kwargs)
    
    def plot_individual(self,*args,**kwargs) -> None:
        for process_object in self.list_of_process_objects:
            process_object.plot_individual(*args,**kwargs)

    def __str__(self) -> str:
        return "".join(["".join(process_object.__str__()) for process_object in self.list_of_process_objects])
    